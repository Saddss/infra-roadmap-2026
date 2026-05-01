# W13: SGLang RadixAttention 笔记

## 核心问题：RadixAttention 解决什么 vLLM PagedAttention 解决不了的问题？

```
（一句话）
```

## 数据结构对比

| | vLLM PagedAttention | SGLang RadixAttention |
|---|---|---|
| 粒度 | block 级（默认 16 token / block） | token 级 |
| 数据结构 | block table（list 索引） | Radix Tree（前缀树） |
| 命中策略 | block hash 匹配 | 前缀树最长匹配 |
| 命中精度 |  |  |
| Eviction |  |  |
| 适合场景 |  |  |

## 我读的核心源码

`sglang/srt/mem_cache/radix_cache.py` 关键函数：

```python
（粘 5-10 行最关键的）
```

## 实验：prefix sharing benchmark

**实验设计**：

固定 system prompt（512 tokens），生成 100 个不同 user query，每个 query 长度 64 tokens。

**结果**：

| 框架 | 第 1 次请求 TTFT | 第 100 次请求 TTFT | 平均 throughput |
|---|---|---|---|
| vLLM (no prefix cache) |  |  |  |
| vLLM (prefix cache) |  |  |  |
| SGLang |  |  |  |

## 我的结论

```
（什么场景下 SGLang 优势明显？什么场景下没区别？）
```
