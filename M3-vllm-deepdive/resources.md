# M3 资源清单 · vLLM 源码

> 收录原则见顶层 [RESOURCES.md](../RESOURCES.md) 的"收录与排序原则"。

## 必读三篇（按顺序）

1. ⭐⭐⭐ **Smarter《vLLM 系统拆解》系列** — [smarter.xin 2025.10-11 持续更新](https://smarter.xin/posts/354d88e4/)
   - **七层结构图是教科书级**，按 vLLM 官方架构文档对齐
   - 面试问到某个能力落在哪层时直接能答（"continuous batching 在哪？" → "scheduler 层"）
2. ⭐⭐⭐ **shijuzhao《vLLM V1: 大模型推理系统的教科书》** — [shijuzhao.github.io 2025.2](https://shijuzhao.github.io/vllm-v1)
   - **基于 v1 0.7.3 版本逐行代码解读，最完整的 V1 中文资料**
   - 从调用接口讲起，一层一层往下解读
   - 重点关注调度器章节（V0 vs V1 最大区别）
3. ⭐⭐⭐ **猛猿《图解大模型计算加速系列：vLLM 源码解析 1-N》** — [微信公众号系列 2024](https://mp.weixin.qq.com/s/r_t6_zMvPT7za82MZX4oRA)
   - 多篇连载，强调"图解 + 整体架构"
   - 图比文字更值钱

## 通俗 + 进阶补充

- ⭐⭐ "Inside vLLM: Anatomy of a High-Throughput LLM Inference System" 中文版 — [知乎 2025](https://zhuanlan.zhihu.com/p/2021773453523522489) — 翻译自 vLLM 官方博客，权威
- ⭐⭐ quant67《大模型基础设施工程》系列 · PagedAttention 与 Continuous Batching — [quant67.com 2025](https://quant67.com/post/llm-infra/12-paged-continuous/12-paged-continuous.html)
  - 含调参手册（`gpu_memory_utilization` / `max_num_batched_tokens` 等）
- ⭐⭐ 《vLLM 核心机密：大模型推理引擎内部长啥样？》— [微信公众号 2025](https://mp.weixin.qq.com/s/SnZVSAwlAtyp01YqKoWIgg) — 解读 vLLM Office Hours #32

## 官方文档

- ⭐⭐⭐ [vLLM Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview.html) — 官方架构文档
- ⭐⭐⭐ [vLLM V1 Guide](https://docs.vllm.ai/en/stable/usage/v1_guide/) — V1 vs V0 feature 矩阵
- ⭐⭐ [vLLM Paper (SOSP'23)](https://arxiv.org/abs/2309.06180) — Kwon et al. 原论文

## 视频（B 站）

- 搜 "vLLM 源码全流程分析"（lyrry1997 系列，带飞书课件）
- 搜 "vLLM 大模型推理框架 分块显存管理"
- 搜 "AI INFRA 学习 LLM 全景图"

## 关键源码文件（W9-W12 必读）

| 文件 | 职责 | 优先级 |
|---|---|---|
| `vllm/v1/engine/core.py` | 控制中枢 / busy loop | ⭐⭐⭐ |
| `vllm/v1/core/sched/scheduler.py` | 调度策略大脑 | ⭐⭐⭐ |
| `vllm/v1/core/kv_cache_manager.py` | KV cache + block table | ⭐⭐⭐ |
| `vllm/v1/worker/gpu_model_runner.py` | forward + sampling 执行 | ⭐⭐⭐ |
| `vllm/v1/engine/llm_engine.py` | 引擎封装层 | ⭐⭐ |
| `vllm/v1/executor/multiproc_executor.py` | worker 进程组织 | ⭐⭐ |
| `vllm/v1/worker/gpu_worker.py` | 单卡设备代理 | ⭐⭐ |
| `csrc/attention/attention_kernels.cu` | PagedAttention CUDA kernel | ⭐ |
| `vllm/v1/sample/sampler.py` | 采样器 | ⭐ |

## 必看 Issue / Discussion

- vLLM 主仓库 `good first issue` 标签（找 PR 起点）
- vLLM 主仓库 `documentation` 标签
- vLLM RFC 系列（搜 v1 RFC）

## SGLang 横向对比（W12 末，为 M4 铺垫）

- ⭐⭐⭐ [SGLang 官方 RadixAttention 文档](https://mintlify.com/sgl-project/sglang/concepts/radix-attention) — 含 `radix_cache.py` 关键源码片段
- ⭐⭐⭐ [LMSYS 官方博客 · Fast and Expressive LLM Inference with RadixAttention and SGLang](https://lmsys.org/blog/2024-01-17-sglang/) — 原作者团队解读
- ⭐⭐《RadixAttention 技术详解：从原理到 SGLang 实践及 vLLM APC 对比》— [华为昇腾开源专区 2026.3](https://ascendai.csdn.net/69c390dd54b52172bc642216.html)

## 三篇博客的完整大纲

→ 见 [`blog-drafts/`](./blog-drafts/) 目录中的三个模板
