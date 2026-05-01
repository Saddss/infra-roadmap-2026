# DPO（Direct Preference Optimization）

> Stanford 团队提出，"绕开 RM 直接对齐"。Llama-3 用了 DPO 的变体。

## 核心思想

不要 RM、不要 critic、不要 RL！直接用偏好对（chosen, rejected）做监督学习。

## Loss 公式

```
L_DPO = -E[(chosen, rejected)]
        log σ(β * (log π_θ(chosen|x) - log π_θ(rejected|x)
                  - log π_ref(chosen|x) + log π_ref(rejected|x)))
```

直觉理解：

```
让 π_θ 把 chosen 概率拉高、rejected 概率拉低
β 控制偏离 π_ref 的程度（越大越保守）
```

## 优势 vs PPO

| | PPO | DPO |
|---|---|---|
| 训练复杂度 | 高（4 个模型） | 低（2 个模型） |
| 训练稳定性 | 中 | **高** |
| RM 需要训吗 | 需要 | **不需要** |
| 效果上限 | 高 | 略低（无法做"在线"探索） |

## 局限

DPO 是"离线"的——只用预先标注好的偏好对，没法像 PPO 那样在训练中实时探索新策略。

## 衍生算法

- **IPO**：把 DPO 的 log σ 改为 (...)²，更稳定
- **cDPO**：处理标签噪声
- **KTO**：用绝对评分代替偏好对（"good/bad"代替 "A 比 B 好"）
- **SimPO**：去掉 ref 模型，进一步简化

## OpenRLHF 中的 DPO

```bash
python train_dpo.py \
  --pretrain Qwen/Qwen2.5-7B \
  --beta 0.1 \
  --dataset openai/summarize_from_feedback ...
```

## 面试可讲

- Q1：DPO 不用 RM 也能对齐吗？数学原理是什么？
- Q2：DPO vs PPO，什么场景选谁？
- Q3：Llama-3 用的是 DPO 还是变体？
- Q4：DPO 的 β 怎么调？
