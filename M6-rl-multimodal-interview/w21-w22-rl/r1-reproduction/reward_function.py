"""
Countdown 任务的奖励函数

任务规则:
    给一组数字 [a, b, c, d] 和一个目标 target，
    用 +、-、*、/ 拼出表达式让结果等于 target

奖励:
    1. 格式奖励：包含 <think>...</think> 和 <answer>...</answer>
    2. 准确性奖励：表达式合法 + 结果正确
"""

import re
from typing import List


def format_reward(completions: List[str], **kwargs) -> List[float]:
    """检查是否符合 R1 输出格式"""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    rewards = []
    for c in completions:
        if re.match(pattern, c.strip(), re.DOTALL):
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


def equation_reward(completions: List[str], target: List[int], nums: List[List[int]], **kwargs) -> List[float]:
    """检查 <answer> 中的表达式是否正确"""
    rewards = []
    for c, t, ns in zip(completions, target, nums):
        m = re.search(r"<answer>(.*?)</answer>", c, re.DOTALL)
        if not m:
            rewards.append(0.0); continue
        expr = m.group(1).strip()

        # 安全检查：只允许数字和运算符
        if not re.match(r"^[\d+\-*/().\s]+$", expr):
            rewards.append(0.0); continue

        # 检查使用的数字
        used = sorted(int(x) for x in re.findall(r"\d+", expr))
        if used != sorted(ns):
            rewards.append(0.0); continue

        try:
            value = eval(expr)
            if abs(value - t) < 1e-5:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


def reward_fn(prompts, completions, target=None, nums=None, **kwargs):
    """主奖励函数：格式 + 准确性"""
    fr = format_reward(completions)
    er = equation_reward(completions, target=target, nums=nums) if target is not None else [0] * len(completions)
    return [f + e for f, e in zip(fr, er)]


if __name__ == "__main__":
    # 测试
    completions = [
        "<think>let me try 19 + 36 - 55 = 0, no wait, the target is 65...</think>\n<answer>19 + 36 + 10</answer>",
        "<answer>just answer</answer>",
        "<think>thinking</think><answer>1 + 1</answer>",
    ]
    target = [65, 65, 2]
    nums = [[19, 36, 10], [19, 36, 10], [1, 1]]
    print("Format rewards:", format_reward(completions))
    print("Equation rewards:", equation_reward(completions, target=target, nums=nums))
    print("Total:", reward_fn(None, completions, target=target, nums=nums))
