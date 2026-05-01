# GRPO（Group Relative Policy Optimization）

> DeepSeekMath 论文提出，用于 R1。这是当下最火的 RL 算法。

## 核心创新：用组平均代替 Critic

传统 PPO：

```
A_t = R - V(s_t)   # advantage = reward - critic 估的 value
```

需要训练 critic（值函数模型）→ 显存翻倍。

GRPO：

```
对同一个 prompt 采样 G 个 response (o_1, ..., o_G)
A_i = (R_i - mean({R_j})) / std({R_j})   # 组归一化
```

→ **不需要 critic** → 显存省一半，训练加速。

## 完整 Loss 公式

```
L_GRPO(θ) = E[
    (1/G) Σ_i (1/|o_i|) Σ_t [
        min(
            ratio_t * A_i,
            clip(ratio_t, 1-ε, 1+ε) * A_i
        ) - β * KL(π_θ || π_ref)
    ]
]

ratio_t = π_θ(o_{i,t} | q, o_{i,:t}) / π_old(o_{i,t} | q, o_{i,:t})
```

## 在 R1 中的应用

```
Stage 2 训练用 GRPO：
- 对每个数学题采样 G=8 个解答
- 准确性奖励 + 格式奖励
- group 内归一化得到 advantage
```

## 与 PPO 的对比

| | PPO | GRPO |
|---|---|---|
| Critic | 需要 | **不需要** |
| 显存 | 4 个模型（policy/ref/RM/critic） | 3 个模型（policy/ref/RM） |
| 计算量 | 高 | 中 |
| 稳定性 | 高 | 中（震荡问题，初七大佬指出） |

## 在 OpenRLHF 中的实现

```python
# OpenRLHF 的 advantage_estimator
--advantage_estimator group_norm  # GRPO

# 同时支持
--advantage_estimator dr_grpo     # 简化版（去掉局部 std 归一化）
--advantage_estimator reinforce_baseline  # REINFORCE++ baseline
```

## 我的训练记录（W22 跑完后填）

```
模型: Qwen2.5-1.5B-Instruct
任务: Countdown
硬件: 3×A800 80GB
时长: __ 小时
最终奖励曲线: （贴图）
```

## 面试可讲

- Q1：GRPO vs PPO？
- Q2：为什么 R1 选 GRPO 不选 DPO？
- Q3：GRPO 在小模型（< 3B）上为什么效果差？
- Q4：vLLM 在 GRPO 训练中扮演什么角色？（rollout engine！）
