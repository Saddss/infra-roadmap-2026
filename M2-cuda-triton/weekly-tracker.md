# M2 周记 · CUDA + Triton 实战

> 每周日 30 分钟填这份。

## W5 周记 · CUDA 基础

**完成度**：

- [ ] PMPP Chapter 1-4 阅读完
- [ ] Vector add CUDA 实现
- [ ] 朴素 matmul CUDA 实现
- [ ] CUDA-MODE Lecture 1-3 看完

**核心洞察**（用自己话写）：

```
（thread / block / grid 是怎么回事？SM 是什么？warp 为什么是 32？）
```

**卡在哪 / 怎么解决**：

| 卡点 | 持续多久 | 怎么解决 |
|---|---|---|
|  |  |  |

**焦虑分**：1-10

---

## W6 周记 · CUDA Tiling

**完成度**：

- [ ] PMPP Chapter 5-6 阅读完
- [ ] Tiled matmul 实现，对比朴素版加速比
- [ ] Tree-based reduction 实现
- [ ] 触发并消除 bank conflict 实验

**核心洞察**：

```
（为什么 shared memory tiling 有效？bank conflict 何时发生？）
```

**Benchmark 数据**：

```
矩阵规模 1024x1024 fp32:
  naive matmul    : ___ ms
  tiled matmul    : ___ ms
  cuBLAS sgemm    : ___ ms
```

**焦虑分**：1-10

---

## W7 周记 · Triton 三件套

**完成度**：

- [ ] Triton vector add 跑通
- [ ] Triton fused softmax 跑通
- [ ] Triton matmul 跑通（含 autotune）
- [ ] BBuf Triton 入门笔记三看完

**核心洞察**：

```
（Triton 的抽象层次比 CUDA 高在哪？block pointer 是什么？为什么不用 thread index？）
```

**焦虑分**：1-10

---

## W8 周记 · FlashAttention V2

**完成度**：

- [ ] FlashAttention V1/V2 论文读完
- [ ] 中文图解看完
- [ ] Triton 实现 FlashAttention V2 forward
- [ ] benchmark vs `F.scaled_dot_product_attention`
- [ ] 博客发布

**博客发布链接**：

```
```

**博客评论数**：__（≥ 3 才能进 M3）

**Benchmark 结果**：

```
seq_len=2048, head_dim=64, num_heads=16:
  naive PyTorch    : ___ ms
  我的 Triton FA2  : ___ ms
  F.SDPA           : ___ ms（参考）
```

**核心洞察**（这一段写在博客里）：

```
（在线 softmax 的更新公式我能否在白板上推 5 分钟？）
```

**M2 总结**（200 字写给一个月后的自己）：

```
```

**焦虑分**：1-10
