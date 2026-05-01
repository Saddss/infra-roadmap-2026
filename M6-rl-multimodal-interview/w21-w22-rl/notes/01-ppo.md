# PPO（Proximal Policy Optimization）

> RLHF 的标杆。GPT-4、Claude、Llama-2 都用 PPO。

## 四个模型组件

```
Actor (Policy)        : 当前在训练的模型
Reference (Ref)       : 训练前的初始模型，用于 KL 惩罚
Reward Model (RM)     : 给 response 打分
Critic (Value)        : 估 state value
```

## 核心 Loss

```
L_PPO = E[ min(ratio * A, clip(ratio, 1-ε, 1+ε) * A) ] - β * KL(π_θ || π_ref)

ratio = π_θ(a|s) / π_old(a|s)
A = R - V(s)         # advantage
```

## clip 的作用

防止 policy 一次更新太大 → 训练崩溃。clip 把 ratio 限制在 [1-ε, 1+ε]（典型 ε=0.2）。

## 实现要点（"PPO 的 N 种实现细节"）

参考 HuggingFace《PPO 的 N 种实现细节》：

1. **Reward 归一化**
2. **Value loss clipping**
3. **Generalized Advantage Estimation (GAE)**
4. **Per-token KL penalty**（OpenRLHF 用这个）
5. **Adam 学习率 warmup**

## OpenRLHF 中的 PPO

```bash
python train_ppo_ray.py \
  --pretrain Qwen/Qwen2.5-7B-Instruct \
  --reward_pretrain ./Qwen2.5-7B-RM \
  --advantage_estimator gae \
  --actor_learning_rate 5e-7 \
  --critic_learning_rate 9e-6 \
  --kl_coef 0.05 \
  --vllm_num_engines 4 \  # 用 vLLM 做 rollout
  --vllm_tensor_parallel_size 1
```

## vLLM 在 PPO 中的角色

```
[Actor 模型 forward] → 生成 response
↓ (这一步特别慢，是 PPO 的瓶颈)

OpenRLHF 的解法: 
  Actor 用 vLLM 做 rollout (10x 加速)
  vLLM engine 和 Actor 共享权重 (hybrid engine)
```

## 面试可讲

- Q1：PPO 为什么有 4 个模型？哪个最大？
- Q2：KL coef 怎么调？太大太小各有什么问题？
- Q3：GAE 的 λ 是什么？
- Q4：为什么 RLHF 训练成本高？vLLM 怎么帮的？
