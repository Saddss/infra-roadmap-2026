# M3 资源清单 · vLLM 源码

## 必读三篇（按顺序）

1. ⭐⭐⭐ [quant67 大模型基础设施工程系列 · PagedAttention 与 Continuous Batching](https://quant67.com/post/llm-infra/12-paged-continuous/12-paged-continuous.html)
   - "土法炼钢兴趣小组"，从 PagedAttention 到 vLLM v1 全覆盖，最完整
2. ⭐⭐⭐ [Smarter's blog · vLLM 系统拆解 · 七层结构](https://smarter.xin/posts/354d88e4/)
   - 按"七层结构"梳理，面试问到落在哪个文件直接答
3. ⭐⭐ [vLLM 底层 PagedAttention 与 Continuous Batching 解释（带源码片段）](https://jishuzhan.net/article/2045034961988812802)
   - 通俗易懂，适合先读

## 官方文档

- [vLLM Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview.html)
- [vLLM Paper](https://arxiv.org/abs/2309.06180)（Kwon et al., SOSP 2023）

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

- vLLM 主仓库 `good first issue` 标签
- vLLM 主仓库 `documentation` 标签
- vLLM RFC 系列（搜 v1 RFC）

## SGLang 横向对比（W12 末）

- [SGLang vs vLLM 2026 深度对比](https://chenxutan.com/d/1513.html)
- [网硕互联 · SGLang vs vLLM 核心差异](https://www.wsisp.com/helps/53774.html)

## 三篇博客的完整大纲

→ 见 [`blog-drafts/`](./blog-drafts/) 目录中的三个模板
