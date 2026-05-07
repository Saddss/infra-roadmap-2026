# M4 资源清单 · 推理框架横向 + 工程优化

> 收录原则见顶层 [RESOURCES.md](../RESOURCES.md) 的"收录与排序原则"。

## SGLang

### 官方资料

- ⭐⭐⭐ [SGLang 官方 RadixAttention 文档](https://mintlify.com/sgl-project/sglang/concepts/radix-attention) — 含 `radix_cache.py` 源码
- ⭐⭐⭐ [LMSYS 官方博客 · Fast and Expressive LLM Inference with RadixAttention and SGLang](https://lmsys.org/blog/2024-01-17-sglang/) — 原作者团队解读
- ⭐⭐ [SGLang Paper (arXiv:2312.07104)](https://arxiv.org/abs/2312.07104)

### 中文解读

- ⭐⭐⭐ 《RadixAttention 技术详解：从原理到 SGLang 实践及 vLLM APC 对比》— [华为昇腾开源专区 2026.3](https://ascendai.csdn.net/69c390dd54b52172bc642216.html)
  - 含 SGLang 源码 + 与 vLLM Automatic Prefix Caching 详细对比
- ⭐⭐ [SGLang Notes（基于 Awesome-ML-SYS-Tutorial）](https://www.liaojiayi.com/blog/sglang) — 简版要点
- ⭐⭐ [SGLang vs vLLM 2026 深度对比](https://chenxutan.com/d/1513.html)

### 关键源码文件

- `sglang/srt/managers/cache_controller.py`
- `sglang/srt/mem_cache/radix_cache.py`
- `sglang/srt/layers/radix_attention.py`

## Mooncake / KV Cache 存储 / PD 分离（**第一作者本人解读，权威性 100%**）

- ⭐⭐⭐ **章明星《Mooncake (1)：在月之暗面做月饼，Kimi 以 KVCache 为中心的分离式推理架构》** — [清华助理教授 + 月之暗面 KVCache.AI 团队负责人 / 2024.6 转载于网易](https://www.163.com/dy/article/J64853CG055689ZC.html)
  - **论文一作自己写的解读**
  - 含原论文未提的 design choice 思考
  - 讲清楚 TBT vs TPOT、Prefill 节点应不应当独立存在等核心问题
- ⭐⭐⭐ 《对话清华章明星、月之暗面许欣然：Mooncake 架构背后》— [硅星人 ACC 2024.11](https://view.inews.qq.com/a/20241121A02FF200)
  - 用"备菜/炒菜"类比讲清 PD 分离
  - 工业落地视角

### Mooncake 论文

- ⭐⭐⭐ [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving (arXiv:2407.00079)](https://arxiv.org/abs/2407.00079)
- ⭐⭐ [Mooncake 阅读笔记 · 调度算法详解](https://juejin.cn/post/7510051346306105394) — 含 PD 分离调度算法逐步解析

### LMCache（多级 KV cache）

- ⭐⭐⭐ [LMCache 官方文档 · Architecture Overview](https://docs.lmcache.ai/developer_guide/architecture.html) — GPU/CPU/Disk/Remote 四级架构图
- ⭐⭐ [LMCache CPU RAM 配置](https://docs.lmcache.ai/kv_cache/cpu_ram.html) — CPU offload 实战
- ⭐⭐ [LMCache + Mooncake 集成](https://docs.lmcache.ai/kv_cache/storage_backends/mooncake.html) — 分布式 KV cache 实战配置

## Speculative Decoding

### 综述

- 知乎搜 "投机采样综述 2025"
- 知乎搜 "Medusa EAGLE MTP 对比"

### 论文

- Medusa: Multiple Decoding Heads
- EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty
- MTP（Multi-Token Prediction）：DeepSeek-V3 论文中的设计

### vLLM 中的实现

- `vllm/spec_decode/` 目录
- vLLM v1 把 spec decode 统一到调度器（不再是插件）

## 量化

### FP8

- 论文：FP8 Training (NVIDIA)
- DeepSeek-V3 报告中的 FP8 训练章节
- vLLM 的 `--quantization fp8`

### AWQ / GPTQ

- AWQ 论文：Activation-aware Weight Quantization
- GPTQ 论文
- 知乎搜 "AWQ vs GPTQ 对比"
- HuggingFace 的 AutoAWQ / AutoGPTQ 库

### INT4 KV Cache

- vLLM 的 `--kv-cache-dtype int4` / `fp8_e5m2`
- 搜 "INT4 KV Cache 实现"

## Disaggregated Prefill / xPyD 弹性 PD

> 完整深度专题见 [`w16-disaggregated-prefill/notes-deep.md`](./w16-disaggregated-prefill/notes-deep.md)

### 核心 PR / RFC（2025-2026）

- ⭐⭐⭐ [vLLM PR #18242 · An native implementation of xPyD based on P2P NCCL](https://github.com/vllm-project/vllm/pull/18242) — vLLM xPyD 核心实现
- ⭐⭐⭐ [vLLM PR #12957 · Support XpYd disaggregated prefill with MooncakeStore](https://github.com/vllm-project/vllm/pull/12957) — Mooncake backend
- ⭐⭐⭐ [vLLM RFC #22799 · ATTN-FFN Disaggregation for MoE Models](https://github.com/vllm-project/vllm/issues/22799) — **2025.8 启动的 AF 分离 RFC，2026 重点跟踪**
- ⭐⭐⭐ [SGLang PD Disaggregation 官方文档（中文）](https://docs.sglang.com.cn/advanced_features/pd_disaggregation.html) — 含完整启动命令
- ⭐⭐ [SGLang Issue #9442 · Configuration of xPyD](https://github.com/sgl-project/sglang/issues/9442) — 配置示例
- ⭐⭐ [SGLang PR #18163 · improve kv offset calculation for MHA model with different tp size](https://github.com/sgl-project/sglang/pull/18163) — 实测 TTFT 优化案例

### Multi-LoRA

- vLLM `vllm/lora/` 目录
- `vllm serve --enable-lora --lora-modules ...`

## AF 分离（Attention-FFN Disaggregation）—— **2026 新方向**

- ⭐⭐⭐ **MegaScale-Infer (字节跳动, arXiv:2504.02263)** — [PDF](https://arxiv.org/pdf/2504.02263v2)
  - 为 MoE 模型提供 attention 和 FFN 分离的并行化策略
  - Ping-pong 流水线 + M2N 通信库
  - **相比 SOTA 提升 1.90× 吞吐**
- ⭐⭐⭐ **Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving (arXiv:2601.21351)** — [PDF](https://arxiv.org/html/2601.21351v1)
  - 给出 AFD 资源配置比例的可解析框架
  - 理论最优 A/F 比例与模拟值差距 ≤10%
- ⭐⭐⭐ **百度 AFD 死区分析 (arXiv:2602.09721)** — [PDF](https://arxiv.org/pdf/2602.09721)
  - "死区"现象：标准集群上增加 FFN 实例数无法改进 FLOPS 利用率
  - AFD 节点级离散扩展比 EP 连续 batch 调整产生更高的不平衡惩罚

## 大 EP / EPLB（**MoE 推理标配**）

- ⭐⭐⭐ [DeepSeek EPLB GitHub](https://github.com/deepseek-ai/EPLB) — DeepSeek 开源 Expert Parallelism Load Balancer
- ⭐⭐⭐ [腾讯云 · EP 架构：DeepSeek 突破性实践背后](https://cloud.tencent.com.cn/developer/article/2504080) — EP 终极形态之争（**必读**）
- ⭐⭐⭐ [腾讯云 · 如何重现 DeepSeek 推理性能突破](https://cloud.tencent.com/developer/article/2523038) — RTP-LLM 阿里云灵骏实测：Prefill 42.6K TPS / Decode 14.7K TPS
- ⭐⭐ [阿里云 · DeepSeek EPLB 冗余专家策略](https://developer.aliyun.com/article/1654261)
- ⭐⭐ [DeepSeek 开源 EPLB 解读](https://deepseek.csdn.net/67fce31ea5baf817cf48e159.html)

## FlashInfer —— **推理 attention kernel 标杆库**

- ⭐⭐⭐ [flashinfer-ai/flashinfer GitHub](https://github.com/flashinfer-ai/flashinfer) — Zihao Ye（FlashAttention "From Online Softmax to FlashAttention" 作者）维护
- ⭐⭐ [FlashInfer 源码级解读：大模型推理的"底层引擎"是怎么炼成的](https://www.yeyulingfeng.com/534685.html)
- 关键能力：稀疏注意力（达密集 90% 带宽）、JIT 自定义 attention、PageAttention、Top-K 采样 kernel
- 用途：vLLM v1 默认 backend 之一、SGLang `--mm-attention-backend fa3`、DeepSeek V4 SGLang 部署 backbone

## TritonServer (NVIDIA)

- 📝 [CUDA 编程基础与 Triton 模型部署实践](https://mp.weixin.qq.com/s/mXwJAAyYanmmWqLgK0FZNg)
  - 注意：这里"Triton" 指的是 NVIDIA TritonServer，不是 OpenAI Triton

## TRT-LLM（你已经有源码）

- `~/sss/project/TensorRT-LLM/` 是已 clone 的仓库
- 重点看 `tensorrt_llm/serve/` 目录（OpenAI 协议兼容）
- 关注：FP8 batched gemm、xQA kernel、Medusa 实现
