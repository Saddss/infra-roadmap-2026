# 博客大纲 #3：vLLM v1 vs v0 架构演进

> 目标读者：用过 vLLM v0 但不熟悉 v1 的人
> 字数目标：5000+ 字
> 必带：1 张架构对比图 + 性能 benchmark + 至少 5 个具体差异点

## 大纲

### 0. 引子（300 字）

写一段："vLLM 在 2024 年底开始了 v1 重构，2025 年逐步 GA。切到 v1 后通常免费送 30% 吞吐。但 v1 到底改了什么？我把两版源码对比读了一遍，这是发现。"

### 1. v0 的痛点（800 字）

- 1.1 调度器是 Python 写的，单线程
- 1.2 调度和 forward 不流水，GPU 经常空闲
- 1.3 prefix caching、speculative decoding、多模态都是后期补丁式塞进来的
- 1.4 多卡通信开销大

### 2. v1 的设计目标（500 字）

- 调度器搬到更高效的执行层
- 零 CPU 阻塞：调度和 forward pipeline
- 统一的 request lifecycle：prefill / decode / preempt / prefix-hit 走同一条路径
- 原生多模态：vision encoder 与 LLM decoder 共享调度
- 更好的 PP/TP 支持

### 3. 核心架构对比（1500 字 · 重头戏）

放一张表（左 v0、右 v1）：

| 维度 | v0 | v1 |
|---|---|---|
| 调度器位置 |  |  |
| 请求生命周期 |  |  |
| Prefix Caching |  |  |
| Spec Decoding |  |  |
| Structured Output |  |  |
| Multi-LoRA |  |  |
| Disaggregated Prefill |  |  |

每行展开 100-200 字解释。

### 4. v1 引入的新概念（800 字）

- 4.1 EngineCore：busy loop 的"心脏"
- 4.2 KVConnector：为分离部署铺路
- 4.3 统一的 ModelRunnerOutput

### 5. 我做的对比 benchmark（800 字）

跑一个 70B 模型在 v0 和 v1 上：

| 指标 | v0 | v1 | 提升 |
|---|---|---|---|
| Throughput | __ | __ | __ |
| TTFT | __ | __ | __ |
| TPOT | __ | __ | __ |

放对比曲线图。

### 6. SGLang RadixAttention 的对比（700 字）

- v1 的 prefix caching 还是 block 级 hash
- SGLang 的 RadixAttention 是 token 级前缀树
- 各自适合的场景：长 prompt 共享 vs 多轮对话 / Agent

### 7. 写在最后：我的两个观察（400 字）

- 观察 1：vLLM 在追赶 SGLang 的部分能力（结构化输出、tool call）
- 观察 2：未来一年 vLLM 的方向是 ___（你预测一下）

### 8. 推荐阅读（200 字）
