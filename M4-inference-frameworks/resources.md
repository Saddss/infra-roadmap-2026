# M4 资源清单 · 推理框架横向 + 工程优化

## SGLang

- 📦 GitHub：[sgl-project/sglang](https://github.com/sgl-project/sglang)
- 📚 [Official Docs](https://docs.sglang.ai)
- 📝 [SGLang vs vLLM 2026 深度对比](https://chenxutan.com/d/1513.html)
- 📝 [网硕互联 · SGLang vs vLLM 核心差异](https://www.wsisp.com/helps/53774.html)

### RadixAttention 核心文件

- `sglang/srt/managers/cache_controller.py`
- `sglang/srt/mem_cache/radix_cache.py`

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

- 📝 [CUDA 编程基础与 Triton 模型部署实践](https://mp.weixin.qq.com/s/mXwJAAyYanmmWqLgK0FZNg)（同名混淆，这是 NVIDIA TritonServer 不是 OpenAI Triton）
- TRT-LLM 文档：你 `~/sss/project/TensorRT-LLM` 已有源码

## TRT-LLM（你已经有源码）

- `~/sss/project/TensorRT-LLM/` 是你已 clone 的仓库
- 重点看 `tensorrt_llm/serve/` 目录（OpenAI 协议兼容）
- 关注：FP8 batched gemm、xQA kernel、Medusa 实现
