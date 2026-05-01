# M3 周记 · vLLM 源码

## W9 周记 · 跑通 + 七层通读

**完成度**：

- [ ] vLLM clone 完，跑通 offline 和 online demo
- [ ] 通读 `engine/core.py`、`sched/scheduler.py`、`worker/gpu_model_runner.py`
- [ ] **第 1 个 PR 提交**（Tier 0：typo / 翻译 / docstring）
- [ ] 在 `source-reading-notes/01-engine-core.md` 等笔记里记录主要数据流

**PR 链接**：

```
PR #__:
```

**核心洞察**：

```
（一个请求从 HTTP 进来，经过哪几个对象 / 函数？画一张时序图）
```

**焦虑分**：1-10

---

## W10 周记 · PagedAttention

**完成度**：

- [ ] 精读 `kv_cache_manager.py`
- [ ] 精读 `csrc/attention/attention_kernels.cu`
- [ ] 画出 block table 工作图（mermaid）
- [ ] 博客《PagedAttention 原理与 Block Table 实现》草稿

**核心洞察**：

```
（block table 解决了什么问题？OS 虚拟内存的类比是什么？）
```

**焦虑分**：1-10

---

## W11 周记 · Continuous Batching

**完成度**：

- [ ] 精读 scheduler.step()
- [ ] 理解 chunked prefill / prefix caching 何时触发
- [ ] 实验：改 `gpu_memory_utilization` (0.5/0.85/0.95) 看抢占行为
- [ ] 实验：改 `max_num_batched_tokens` 画 throughput-latency 曲线
- [ ] 博客《vLLM 一个请求的一生》草稿

**实验结果**：

```
（贴吞吐-延迟曲线截图链接）
```

**焦虑分**：1-10

---

## W12 周记 · PR + 收尾

**完成度**：

- [ ] **第 2 个 PR 提交**（Tier 1/2：补单测 / 修小 bug）
- [ ] 三篇博客全部发布
- [ ] SGLang RadixAttention 设计读完，写在博客《v1 vs v0》中作为对比

**PR #2 链接**：

```
PR #__:
```

**博客链接**：

```
1. PagedAttention：
2. 一个请求的一生：
3. v1 vs v0：
```

**M3 总结**（200 字写给一个月后的自己）：

```
```

**焦虑分**：1-10
