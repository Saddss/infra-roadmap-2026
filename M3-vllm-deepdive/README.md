# M3（W9-W12）：vLLM 源码 —— 你的最大杀招

## 目标

能在 1 小时内向面试官完整讲清"一个请求从 HTTP 进入到 token 流出来"的全链路；至少向 vLLM 主仓库提交 **2-3 个 Tier 0/1 PR**（doc / 测试 / typo）+ 写完三篇深度博客。

## 必交产出物

- [ ] **3 篇深度博客**：
  - 《vLLM 一个请求的一生》
  - 《PagedAttention 原理与 Block Table 实现》
  - 《vLLM v1 vs v0 架构演进》
- [ ] **2-3 个 Tier 0/1 PR**（看 `pr-tracker/` 模板）
- [ ] **5 份源码解读笔记**（看 `source-reading-notes/`）

## 每周拆解

### W9：跑通 + 七层结构通读

- [ ] clone vllm-project/vllm，checkout 最新 stable tag
- [ ] 跑通 offline + online 两个 demo
- [ ] 通读三个核心文件（带笔记到 `source-reading-notes/`）：
  - `vllm/v1/engine/core.py`
  - `vllm/v1/core/sched/scheduler.py`
  - `vllm/v1/worker/gpu_model_runner.py`
- [ ] **Tier 0 PR #1**：找一个 typo / 翻译一段 doc 提 PR

### W10：PagedAttention 精读

- [ ] 精读 `vllm/v1/core/kv_cache_manager.py`
- [ ] 精读 `csrc/attention/` 中的 paged_attention_kernel
- [ ] 画出 block table 工作图（用 mermaid）
- [ ] 写博客《PagedAttention 原理与 Block Table 实现》草稿

### W11：Continuous Batching + Prefix Caching

- [ ] 精读 scheduler.step() 完整逻辑
- [ ] 理解 chunked prefill / prefix caching
- [ ] 做实验：改 `gpu_memory_utilization` 和 `max_num_batched_tokens`
- [ ] 输出 throughput-latency 曲线（脚本在 `experiments/`）
- [ ] 写博客《vLLM 一个请求的一生》草稿

### W12：提交 PR + 写完所有博客

- [ ] 找一个 `good first issue` 或 `bug` 标签的 issue
- [ ] **Tier 1/2 PR #2**（补单测 / 修小 bug）
- [ ] 写博客《vLLM v1 vs v0 架构演进》
- [ ] 同时读 SGLang 的 RadixAttention 设计

## 阶段切换检查表（进 M4 前）

- [ ] vLLM repo 已 fork 并 clone 到本地
- [ ] 至少 2 个 PR 已提交（不要求 merge）
- [ ] 3 篇博客已发布（每篇 ≥ 5000 字带源码片段）
- [ ] `weekly-tracker.md` 4 周记录都填了
- [ ] 简历更新："深度阅读 vLLM v1 引擎、调度器、PagedAttention 核心模块；提交 X 个开源 PR"

## 资料

→ 见 [`resources.md`](./resources.md)

## 周记

→ 见 [`weekly-tracker.md`](./weekly-tracker.md)

## PR 跟踪

→ 见 [`pr-tracker/`](./pr-tracker/)

## 源码笔记

→ 见 [`source-reading-notes/`](./source-reading-notes/)
