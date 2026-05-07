# 08 · 三大训练框架横向对比 · FSDP vs DeepSpeed vs Megatron-Core

> W17-W18 进阶。学完笔记 01-07 单点知识后，看这一份建立"该选哪个框架"的全局判断。

## 0. 三大流派 · 一句话定位

| 流派 | 一句话定位 | 哲学 |
|---|---|---|
| **FSDP / FSDP2**（PyTorch 原生） | "代码侵入最少" | 一切回归 PyTorch nn.Module，AI 友好 |
| **DeepSpeed**（Microsoft） | "JSON 配一切" | ZeRO + Offload，最少改模型代码 |
| **Megatron-Core**（NVIDIA） | "MFU 最高" | 大模型量身定制，工业级 |

## 1. 详细对比表

### 1.1 设计哲学

| 维度 | FSDP/FSDP2 | DeepSpeed | Megatron-Core |
|---|---|---|---|
| 代码侵入 | 极低（直接包裹 nn.Module） | 低（JSON 配 + 改 trainer） | 高（必须改 layer 实现） |
| 调试门槛 | 低（PyTorch 原生工具链） | 中 | 高（NCCL/PTX 出错难追） |
| TP 支持 | 较晚（FSDP2 才有） | 弱（不如 Megatron） | **极强**（首创） |
| PP 支持 | 弱 | 中（PipelineModule） | 极强 |
| EP / MoE 支持 | 弱 | 强（DeepSpeed-MoE） | 强（Megatron-Core MoE） |
| Offload 能力 | 弱 | **极强**（CPU + NVMe + Infinity） | 弱 |
| MFU（A100/H100） | 30-45% | 35-45% | **50%+** |
| 生态 | 与 HF Trainer 原生 | HF Accelerate 集成 | HF 集成弱，需自己改 |

### 1.2 适用规模建议

| 模型规模 | 推荐方案 | 备注 |
|---|---|---|
| < 1B | **FSDP2** 或简单 DDP | 单机够用，DeepSpeed 反而麻烦 |
| 1B - 10B | **FSDP2 + 可选 TP=2/4** 或 Megatron TP=2/4 | 看团队熟悉度 |
| 10B - 70B | **Megatron-Core (TP+PP+DP)** 主流 | 工业界标配，MFU 最高 |
| > 100B / MoE | **Megatron-Core + 自研补丁** 或 **DeepSpeed-MoE** | 千卡+ 训练 |
| Offload 需要 | **DeepSpeed ZeRO-Infinity** 唯一选择 | 比如单机榨干 70B |

### 1.3 显存优化能力

| 优化 | FSDP | DeepSpeed | Megatron |
|---|---|---|---|
| ZeRO-1（切 optimizer states） | ✅ FSDP_NO_SHARD | ✅ stage 1 | ❌（用 distributed optimizer 替代） |
| ZeRO-2（+ 切 gradients） | ✅ FSDP_SHARD_GRAD_OP | ✅ stage 2 | ❌ |
| ZeRO-3（+ 切 parameters） | ✅ FSDP_FULL_SHARD（默认） | ✅ stage 3 | 通过 TP 间接实现 |
| CPU Offload | ✅ | ✅ | ❌ |
| NVMe Offload | ❌ | ✅ ZeRO-Infinity | ❌ |
| Activation Recomputation | ✅ | ✅ | ✅ |
| 选择性 Activation Recomputation | ❌ | ✅ | ✅ |

## 2. ZeRO vs FSDP 关键差异（**面试常问**）

ZeRO-3 ≈ FSDP（思路一样，DeepSpeed 先实现的）。但有几个重要差异：

### 2.1 精度处理

- **DeepSpeed**：强制将优化器参数保持为 FP32
- **FSDP**：默认保持模型加载精度（如 BF16）

→ **少 GPU 数时 DeepSpeed 内存消耗可能比 FSDP 多 2 倍**（因为强制 FP32 优化器）

📚 来源：[HuggingFace · 从 DeepSpeed 到 FSDP 再回到 Accelerate](https://huggingface.co/blog/zh/deepspeed-to-fsdp-and-back)

### 2.2 通信策略

- **DeepSpeed**：reduce-scatter + all-gather（逐层）
- **FSDP**：可配置（FULL_SHARD = ZeRO-3，SHARD_GRAD_OP = ZeRO-2）

### 2.3 Accelerate 0.30+ 已经对齐两者

HF Accelerate 现在支持 FSDP 的两种模式：
- 低精度模式（内存受限场景）
- 混合精度模式（与 DeepSpeed 一致）

## 3. 真实大模型用了什么

| 模型 | 训练框架 | 来源 |
|---|---|---|
| GPT-3 | Megatron-DeepSpeed | OpenAI 论文 |
| BLOOM | Megatron-DeepSpeed | BigScience 论文 |
| OPT | Megatron-LM | Meta 论文 |
| LLaMA-1/2/3 | Megatron 变体 | Meta 论文 |
| Qwen 系列 | Megatron-LM 早期 → 自研 | Qwen 技术报告 |
| DeepSeek 早期 | Megatron-DeepSpeed | DeepSeek 论文 |
| DeepSeek V3/V4 | **自研 + DualPipe + Muon** | V3/V4 技术报告 |

**结论**：**没有银弹**，大厂都是 Megatron 起步 + 自己改。

## 4. 推理 vs 训练框架的关系

```
推理:    vLLM / SGLang / TRT-LLM
            ↑ 基于
         FlashAttention / FlashInfer kernels
            ↑ 基于
         CUDA / Triton

训练:    Megatron-Core / DeepSpeed / FSDP
            ↑ 基于
         PyTorch + NCCL
            ↑ 基于
         CUDA / cuBLAS
```

**关键：训练和推理的框架是两套，不要混淆**。但有重叠：

- vLLM 在 RLHF 场景下作为 rollout engine 进入训练流水线（OpenRLHF、veRL）
- Megatron 在长上下文训练时也用 FlashAttention
- TP/PP/EP 这些并行策略训练推理共享思想，但实现细节不同（推理无反向传播）

## 5. 怎么选（决策树）

```
问题 1: 你的模型多大？
  < 1B → FSDP2，结束
  1B-10B → 问题 2
  10B-70B → Megatron-Core，结束
  > 100B → 你需要专家咨询，不在这个决策树里

问题 2: 你最在意什么？
  代码改最少 → FSDP2
  Offload 到 NVMe → DeepSpeed ZeRO-Infinity
  MFU 最高 → Megatron-Core
  HF 生态最方便 → FSDP2 + Accelerate
```

## 6. 实操建议

### 6.1 学习阶段：先 FSDP，再 Megatron

```bash
# Step 1: 用 HF Accelerate + FSDP 跑通一个小模型（Qwen2.5-1.5B）
accelerate config  # 选 FSDP
accelerate launch train.py

# Step 2: 用 Megatron-LM 跑通 4 卡训练（W17 任务）
torchrun --nproc-per-node=4 pretrain_gpt.py \
  --tensor-model-parallel-size 4 --sequence-parallel ...

# Step 3: 用 DeepSpeed 体验 ZeRO-Infinity（W17 进阶）
deepspeed train.py --deepspeed --deepspeed_config ds_config.json
```

### 6.2 生产阶段：Megatron-Core + 自研补丁

如果你将来在大厂，看到的多半是 "Megatron-Core fork + 公司自定义 layer + 自定义通信 op"。

## 7. 资料

### 必读

- ⭐⭐⭐ [quant67 · Megatron-LM 与 DeepSpeed](https://quant67.com/post/llm-infra/07-megatron-deepspeed/07-megatron-deepspeed.html) — 最完整的中文对比，含三流派演进历史
- ⭐⭐⭐ [HuggingFace · 从 DeepSpeed 到 FSDP 再回到 Accelerate](https://huggingface.co/blog/zh/deepspeed-to-fsdp-and-back) — 官方实战，含两者精度处理差异
- ⭐⭐ [HuggingFace · FSDP vs DeepSpeed 概念指南](https://hugging-face.cn/docs/accelerate/concept_guides/fsdp_and_deepspeed)

### 进阶

- ⭐⭐ [大模型并行训练指南：通俗理解 Megatron-DeepSpeed](https://blog.csdn.net/v_JULY_v/article/details/132462452) — v_JULY_v 写的长文
- ⭐⭐ [Distributed Training: DeepSpeed ZeRO 1/2/3 + Accelerate, Megatron-LM](https://www.cnblogs.com/forhheart/p/18401234)

### 论文

- ⭐⭐⭐ ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (SC'20)
- ⭐⭐⭐ Megatron-LM (arXiv:1909.08053)
- ⭐⭐⭐ Megatron-LM 2 (arXiv:2104.04473) — 3D 并行
- ⭐⭐⭐ PyTorch FSDP (VLDB'24)

## 8. 面试可讲

### Q1：FSDP 和 DeepSpeed ZeRO-3 区别？

A：
- 思路相同（参数 / 梯度 / 优化器状态分片）
- 实现差异：DeepSpeed 强制 FP32 optimizer，FSDP 默认匹配模型精度
- 生态差异：FSDP 与 HF Trainer 原生集成，DeepSpeed 需 Accelerate 桥接
- TP/PP 配合：DeepSpeed 弱，FSDP2 也弱，都不如 Megatron

### Q2：Megatron 为什么 MFU 高？

A：
- TP 通信优化（NVLink + WGMMA）
- 1F1B Interleaved 调度，bubble 极小
- Selective Activation Recomputation（只重算 attention 的 softmax，不重算 matmul）
- Sequence Parallel 减激活内存

### Q3：什么场景必须 DeepSpeed？

A：
- NVMe Offload 是 DeepSpeed 独家
- 单机训 70B 模型时唯一选择（M5 上你会做这个实验）
- DeepSpeed-MoE 在 MoE 场景仍有优势（虽然 Megatron-Core 也有 MoE）

### Q4：DualPipe 是什么？

A：
- DeepSeek V3 自创的双向 pipeline 调度
- 把 forward 和 backward 同时双向流动，bubble 进一步降低
- 参考 V3 / V4 技术报告
