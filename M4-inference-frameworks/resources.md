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

## Disaggregated Prefill / Multi-LoRA

- vLLM `kv_connector` 接口
- 搜 "vLLM disaggregated prefill 设计"
- vLLM Multi-LoRA: `vllm/lora/`

## TritonServer (NVIDIA)

- 📝 [CUDA 编程基础与 Triton 模型部署实践](https://mp.weixin.qq.com/s/mXwJAAyYanmmWqLgK0FZNg)
  - 注意：这里"Triton" 指的是 NVIDIA TritonServer，不是 OpenAI Triton

## TRT-LLM（你已经有源码）

- `~/sss/project/TensorRT-LLM/` 是已 clone 的仓库
- 重点看 `tensorrt_llm/serve/` 目录（OpenAI 协议兼容）
- 关注：FP8 batched gemm、xQA kernel、Medusa 实现
