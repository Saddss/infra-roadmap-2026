# 项目 #3：vLLM 源码精读 + 开源贡献

## 一句话简介

> "我用 4 周时间把 vLLM v1 的引擎、调度器、PagedAttention 三大核心模块通读了一遍，
>  期间向主仓库提交了 X 个 PR（Y 个已 merge），并写了 3 篇技术博客发在知乎。"

## STAR

### Situation

我之前用 vLLM 都是当 API 工具用，能起 server 但不知道里面发生了什么。
面试时被追问"PagedAttention 在哪个文件实现的"我就答不上来。

### Task

把 vLLM v1 的核心模块通读，能讲清"一个请求从 HTTP 进来到 token 出去"的全链路，
并通过提交 PR 证明我"真的读了"。

### Action

#### A1: 按"七层结构"通读核心文件

参考 Smarter's blog 的七层架构图，我按以下顺序读：

1. `vllm/v1/engine/core.py` — 控制中枢的 busy loop
2. `vllm/v1/core/sched/scheduler.py` — 调度策略
3. `vllm/v1/core/kv_cache_manager.py` — block table 管理
4. `vllm/v1/worker/gpu_model_runner.py` — forward 执行
5. `csrc/attention/attention_kernels.cu` — PagedAttention CUDA kernel

每个文件读完，我画一张数据流图 + 写一份笔记。

#### A2: 三个跑通实验

- 实验 1：改 `gpu_memory_utilization` (0.5/0.85/0.95)，观察抢占行为
- 实验 2：改 `max_num_batched_tokens` (1K/4K/16K)，画 throughput-latency 曲线
- 实验 3：开关 `enable_prefix_caching`，量化 prefix 共享收益

#### A3: 三个 PR

- PR #__: <Tier 0 typo 修复 / 文档改进>
- PR #__: <Tier 1 单测补充>
- PR #__: <Tier 2 小 bug 修复 / sampler 改进>

#### A4: 写 3 篇博客

每篇 5000+ 字、带源码片段、带 mermaid 图：

1. 《vLLM 一个请求的一生》— 端到端时序图 + 全链路解析
2. 《PagedAttention 原理与 Block Table 实现》— OS 类比 + CUDA kernel 解读
3. 《vLLM v1 vs v0 架构演进》— 横向对比 + 我跑的 benchmark

### Result

- 3 个 PR 提交（Y 个 merge）
- 5 份源码笔记，3 篇博客，知乎累计 ___ 赞
- 简历可写："深度阅读 vLLM v1 引擎、调度器、PagedAttention 核心模块；提交 X 个开源 PR 已合并 Y 个"

## 反向追问准备

### 可能被问的难题

**Q：vLLM v1 的 scheduler.step() 大概多少行？说说主要做几件事**
A：约 200 行，核心做 5 件事：
1. 先 schedule 已在 running 的请求
2. 检查 token budget
3. 从 waiting 队列拉新请求做 prefill
4. 处理 chunked prefill
5. 决定是否需要抢占

**Q：PagedAttention kernel 怎么处理非连续访问？性能损失多少？**
A：通过 block_table 跳转访问，每次访问 K[h, block_idx] 时先查 block_table。
   实测比朴素 FlashAttention 慢 < 5%，但显存利用率提升 3-5x。

**Q：vLLM 的 prefix caching 怎么实现？block 级 hash 是怎么做的？**
A：把每个 block 的 token_ids tuple 做 hash，相同的 hash 共享物理 block。
   Eviction 用 LRU。和 SGLang 的 token 级 RadixAttention 比，命中粒度粗但实现简单。

**Q：你那个 PR 改了什么？维护者怎么 review 的？**
A：（具体讲你那个 PR 的 diff 内容 + review 过程）

### 我自己暴露过的弱点

- 弱点：CUDA kernel 部分我读得不够深，PagedAttention kernel 的 warp 级优化没完全吃透
- 怎么解：W18 期间补了 PMPP 的 warp scheduling 章节，下次能讲清楚

## 项目链接

- GitHub fork: https://github.com/<你>/vllm
- PR 列表: <PR 链接>
- 博客 1: <知乎链接>
- 博客 2: <知乎链接>
- 博客 3: <知乎链接>
