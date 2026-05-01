# 100+ 面试题自答

把高频题分类放在这里，每题都自己写一遍答案。

## 文件组织

```
01-transformer-attention.md     # MHA/MQA/GQA/MLA/DSA 全家桶
02-position-encoding-norm.md    # RoPE / RMSNorm / pre-norm vs post-norm
03-cuda-triton-flashattn.md     # CUDA / Triton / FlashAttention
04-vllm-sglang-trtllm.md        # 推理框架
05-quantization-spec-decode.md  # 量化 / 投机采样 / Multi-LoRA
06-parallelism-megatron.md      # TP/PP/DP/EP/CP/SP / FSDP
07-rl-rlhf-grpo.md              # PPO/DPO/GRPO/REINFORCE++
08-architectures-2026.md        # V4 / Qwen3.6 / Kimi / MiniMax
09-multimodal-vit.md            # ViT / Qwen2.5-VL / DiT
10-coding-handwriting.md        # 手撕代码题（attention/RoPE/RMSNorm）
11-system-design.md             # 系统设计题
12-behavioral.md                # 行为面试题
```

## 答题模板

每道题用这个结构：

```markdown
### Q: 问题
来源: 字节一面 / 网上 / 自创

#### 标准答案
（5-10 句话，包含核心机制 + 代价 + 场景）

#### 进阶追问
- Q1: ?
- A1: ?
- Q2: ?
- A2: ?

#### 我的实战例子
（用我 6 个月做的项目里的具体代码 / 实验数据来佐证）
```

## 标准是

每题答案必须能"3 分钟讲清"。能讲清 = 真懂。
