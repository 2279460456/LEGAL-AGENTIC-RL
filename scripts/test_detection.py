"""
Test detection logic for RL training
Run: python -m scripts.test_detection
"""
import sys
sys.path.insert(0, '.')

from src.rl.grpo import GRPOConfig, GRPOTrainer

class TestTrainer(GRPOTrainer):
    pass

trainer = TestTrainer.__new__(TestTrainer)
trainer.config = GRPOConfig()

# Test template residue detection
test_texts = [
    ("作案动机是什么？", False, "正常提问"),
    ("被告人犯诈骗罪，判处有期徒刑三年", False, "正常判决"),
    ("请以你与被告人的身份进行对话", True, "角色指令"),
    ("[法官]: 公诉机关指控", True, "角色格式"),
    ("【你的任务】根据案件信息", True, "任务指令"),
    ("丰台法院认为被告人...", True, "法院名称"),
    ("（一）2013年9月", True, "括号编号"),
    ("公诉机关指控北京市丰台区人民检察院指控并经本院审理查明：一、盗窃犯罪事实被告人杨某伙同张某为给张某偿还债务，在明知不能办理银行按揭贷款的情况下，于2013年1月6日虚构购房合同", True, "完整案情(长)"),
]

print("Template residue detection:")
passed = 0
for text, expected, desc in test_texts:
    result = trainer._has_template_residue(text) or len(text) > 200
    status = "OK" if result == expected else "FAIL"
    if result == expected:
        passed += 1
    print("  [%s] %s (len=%d)" % (status, desc, len(text)))
    print("        detected=%s" % result)

print("\nPassed: %d/%d" % (passed, len(test_texts)))