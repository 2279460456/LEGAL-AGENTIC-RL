"""
SFT Training Module
Supervised fine-tuning for legal role models using LoRA/QLoRA.
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from transformers.utils.logging import set_verbosity_info

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer
from datasets import Dataset
import yaml


@dataclass
class ModelConfig:
    """Model loading configuration"""
    base_model: str = "Qwen/Qwen3-8B"
    use_4bit: bool = False
    use_double_quant: bool = True
    quant_type: str = "nf4"


@dataclass
class LoRAConfig:
    """LoRA adapter configuration"""
    r: int = 64
    lora_alpha: int = 16
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_dropout: float = 0.05
    bias: str = "none"


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    max_seq_length: int = 2048
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    output_dir: str = "models/sft_checkpoint"
    run_name: str = "sft_qwen8b_legal"


@dataclass
class DataConfig:
    """Data configuration"""
    train_data_path: str = "data/processed"
    num_samples: Optional[int] = None
    roles: List[str] = field(default_factory=lambda: ["prosecutor", "defender", "judge"])


def load_config_from_yaml(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_model_config(yaml_config: Dict) -> ModelConfig:
    """Create ModelConfig from YAML config"""
    model_section = yaml_config.get("model", {})
    return ModelConfig(
        base_model=model_section.get("base_model", "Qwen/Qwen3-8B"),
        use_4bit=model_section.get("use_4bit", False),
        use_double_quant=model_section.get("use_double_quant", True),
        quant_type=model_section.get("quant_type", "nf4"),
    )


def create_lora_config(yaml_config: Dict) -> LoRAConfig:
    """Create LoRAConfig from YAML config"""
    lora_section = yaml_config.get("lora", {})
    return LoRAConfig(
        r=lora_section.get("r", 64),
        lora_alpha=lora_section.get("lora_alpha", 16),
        target_modules=lora_section.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        lora_dropout=lora_section.get("lora_dropout", 0.05),
        bias=lora_section.get("bias", "none"),
    )


def create_training_config(yaml_config: Dict) -> TrainingConfig:
    """Create TrainingConfig from YAML config"""
    training_section = yaml_config.get("training", {})
    output_section = yaml_config.get("output", {})
    return TrainingConfig(
        learning_rate=training_section.get("learning_rate", 2e-4),
        batch_size=training_section.get("batch_size", 4),
        gradient_accumulation_steps=training_section.get("gradient_accumulation_steps", 4),
        num_epochs=training_section.get("num_epochs", 3),
        max_seq_length=training_section.get("max_seq_length", 2048),
        warmup_ratio=training_section.get("warmup_ratio", 0.03),
        weight_decay=training_section.get("weight_decay", 0.01),
        logging_steps=training_section.get("logging_steps", 10),
        save_steps=training_section.get("save_steps", 100),
        save_total_limit=training_section.get("save_total_limit", 3),
        output_dir=output_section.get("output_dir", "models/sft_checkpoint"),
        run_name=output_section.get("run_name", "sft_qwen8b_legal"),
    )


def create_data_config(yaml_config: Dict) -> DataConfig:
    """Create DataConfig from YAML config"""
    data_section = yaml_config.get("data", {})
    return DataConfig(
        train_data_path=data_section.get("train_data_path", "data/processed"),
        num_samples=data_section.get("num_samples"),
        roles=data_section.get("roles", ["prosecutor", "defender", "judge"]),
    )


def load_model(model_config: ModelConfig):
    """
    Load base model with optional 4-bit quantization.

    Args:
        model_config: Model configuration

    Returns:
        model, tokenizer
    """
    print(f"\n{'='*60}")
    print(f"Loading model: {model_config.base_model}")
    print(f"{'='*60}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.base_model,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization config (QLoRA)
    if model_config.use_4bit:
        print(f"Using 4-bit quantization (QLoRA): {model_config.quant_type}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_config.quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=model_config.use_double_quant,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
    else:
        print("Using full precision (standard LoRA)")
        model = AutoModelForCausalLM.from_pretrained(
            model_config.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    print(f"Model loaded. Device map: {model.hf_device_map}")
    return model, tokenizer


def setup_lora(model, lora_config: LoRAConfig):
    """
    Setup LoRA adapters on the model.

    Args:
        model: Base model
        lora_config: LoRA configuration

    Returns:
        model with LoRA adapters
    """
    print(f"\n{'='*60}")
    print("Setting up LoRA adapters")
    print(f"{'='*60}")
    print(f"  Rank (r): {lora_config.r}")
    print(f"  Alpha: {lora_config.lora_alpha}")
    print(f"  Target modules: {lora_config.target_modules}")
    print(f"  Dropout: {lora_config.lora_dropout}")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        target_modules=lora_config.target_modules,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model


def load_training_data(data_config: DataConfig) -> Dataset:
    """
    Load and format training data.

    Args:
        data_config: Data configuration

    Returns:
        HuggingFace Dataset
    """
    print(f"\n{'='*60}")
    print("Loading training data")
    print(f"{'='*60}")

    data_path = Path(data_config.train_data_path)
    all_samples = []

    # Load data for each role
    for role in data_config.roles:
        role_file = data_path / f"{role}.json"
        if role_file.exists():
            with open(role_file, "r", encoding="utf-8") as f:
                role_data = json.load(f)
                print(f"  {role}: {len(role_data)} samples")
                all_samples.extend(role_data)
        else:
            print(f"  Warning: {role_file} not found")

    # Also check for combined file
    all_roles_file = data_path / "all_roles.json"
    if all_roles_file.exists() and not all_samples:
        with open(all_roles_file, "r", encoding="utf-8") as f:
            all_samples = json.load(f)
            print(f"  Loaded from all_roles.json: {len(all_samples)} samples")

    # Limit samples if specified
    if data_config.num_samples and len(all_samples) > data_config.num_samples:
        print(f"  Limiting to {data_config.num_samples} samples")
        all_samples = all_samples[:data_config.num_samples]

    print(f"  Total samples: {len(all_samples)}")

    # Convert to dataset
    dataset = Dataset.from_list(all_samples)

    return dataset


def format_instruction(sample: Dict) -> str:
    """
    Format sample into instruction format for training.

    Args:
        sample: Data sample with instruction, input, output

    Returns:
        Formatted text string
    """
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output = sample.get("output", "")

    # Qwen chat format
    if input_text:
        prompt = f"<|im_start|>user\n{instruction}\n\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
    else:
        prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"

    return prompt


def train_sft(
    config_path: str = "configs/sft_config.yaml",
    resume_from_checkpoint: Optional[str] = None,
):
    """
    Main SFT training function.

    Args:
        config_path: Path to YAML config file
        resume_from_checkpoint: Path to checkpoint to resume from
    """
    # Load config
    print(f"\n{'='*60}")
    print("SFT Training - Legal Agentic RL")
    print(f"{'='*60}")
    print(f"Config: {config_path}")

    yaml_config = load_config_from_yaml(config_path)

    model_config = create_model_config(yaml_config)
    lora_config = create_lora_config(yaml_config)
    training_config = create_training_config(yaml_config)
    data_config = create_data_config(yaml_config)

    # Load model
    model, tokenizer = load_model(model_config)

    # Setup LoRA
    model = setup_lora(model, lora_config)

    # Load data
    dataset = load_training_data(data_config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=training_config.output_dir,
        run_name=training_config.run_name,
        num_train_epochs=training_config.num_epochs,
        per_device_train_batch_size=training_config.batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        warmup_ratio=training_config.warmup_ratio,
        weight_decay=training_config.weight_decay,
        logging_steps=training_config.logging_steps,
        save_steps=training_config.save_steps,
        save_total_limit=training_config.save_total_limit,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        max_seq_length=training_config.max_seq_length,
        formatting_func=format_instruction,
        packing=False,
    )

    # Train
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save
    print(f"\n{'='*60}")
    print("Saving model...")
    print(f"{'='*60}")

    output_dir = Path(training_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA adapters
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"Model saved to: {output_dir}")
    print("\nTraining complete!")


def merge_lora_and_save(
    base_model: str,
    lora_path: str,
    output_path: str,
):
    """
    Merge LoRA adapters with base model and save full model.

    Args:
        base_model: Base model path
        lora_path: LoRA adapter path
        output_path: Output path for merged model
    """
    from peft import PeftModel

    print(f"\n{'='*60}")
    print("Merging LoRA with base model")
    print(f"{'='*60}")

    # Load base model
    print(f"Loading base model: {base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Load LoRA
    print(f"Loading LoRA: {lora_path}")
    model = PeftModel.from_pretrained(base, lora_path)

    # Merge
    print("Merging adapters...")
    model = model.merge_and_unload()

    # Save
    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print("Merge complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SFT Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sft_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge LoRA and save full model after training",
    )
    parser.add_argument(
        "--merge-output",
        type=str,
        default="models/sft_merged",
        help="Output path for merged model",
    )

    args = parser.parse_args()

    # Train
    train_sft(
        config_path=args.config,
        resume_from_checkpoint=args.resume,
    )

    # Optionally merge
    if args.merge:
        yaml_config = load_config_from_yaml(args.config)
        model_config = create_model_config(yaml_config)
        training_config = create_training_config(yaml_config)

        merge_lora_and_save(
            base_model=model_config.base_model,
            lora_path=training_config.output_dir,
            output_path=args.merge_output,
        )