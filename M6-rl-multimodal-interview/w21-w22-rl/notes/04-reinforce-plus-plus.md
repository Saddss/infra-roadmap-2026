# REINFORCE++ 

> OpenRLHF 团队提出，被多个 R1 复现项目采用。

## 核心思想

把 PPO 的 trick 加到 REINFORCE 上，去掉 critic。

```
A_t = R - mean(R)         # baseline 用一批 reward 的均值
然后用 PPO 的 ratio + clip：
L = E[ min(ratio * A, clip(ratio, 1-ε, 1+ε) * A) ] - β * Σ_t KL(π_θ || π_ref)
```

注意：KL 是**逐 token**算的（不是整段），这个细节很重要。

## 与 GRPO 的对比

| | GRPO | REINFORCE++ |
|---|---|---|
| baseline | group 内 mean ± std | batch 内 mean |
| KL 处理 | 整段一个 KL | 每个 token 一个 KL |
| 稳定性 | 中（震荡） | **高** |
| 速度 | 慢（每 prompt 采样 G 个） | 快 |
| 多个 R1 复现选用 | 部分 | 多数 |

## 为什么 REINFORCE++ 比 GRPO 稳

- GRPO 的 group 归一化对 reward 噪声敏感
- REINFORCE++ 的 batch baseline 平均效应更稳
- 逐 token KL 让模型不会在某些位置突然偏离 ref

## 在 OpenRLHF 中

```bash
--advantage_estimator reinforce              # REINFORCE++
--advantage_estimator reinforce_baseline    # REINFORCE++-baseline (RLVR 推荐)
```

## 推理任务（RLVR）的最佳实践

OpenRLHF 推荐：

```
RLVR (Reasoning with Verifiable Rewards) 类任务（数学、代码）:
  → reinforce_baseline > GRPO > DPO
```

## 面试可讲

- Q1：REINFORCE++ 怎么稳定 RL 训练？
- Q2：什么是 RLVR？和 RLHF 区别？
- Q3：为什么数学任务用 REINFORCE++ baseline 比 GRPO 好？
