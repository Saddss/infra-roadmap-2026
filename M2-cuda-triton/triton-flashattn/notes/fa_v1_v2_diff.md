# FlashAttention V1 vs V2 关键差异（W8 学习笔记模板）

> 这一份是博客写作的草稿，看完中文图解 + 论文后填这里。

## 1. V1 的核心思想

### 1.1 问题：标准 attention 慢在哪里？

- attention 矩阵 N×N 大小，N=8192 时仅 attn 矩阵就要 ___ MB
- 频繁的 HBM 读写导致 memory-bound
- 公式：`O = softmax(QK^T / √d) @ V`，中间 P=softmax(...) 必须落 HBM

### 1.2 V1 的解法：tiling + online softmax

```
Q 在内循环（i 维度）
K, V 在外循环（j 维度）
每次只算 O[i] 这一小块
```

#### Online softmax 公式：

```
m_new = max(m_old, max(qk_block))
l_new = exp(m_old - m_new) * l_old + sum(exp(qk_block - m_new))
```

为什么需要 m_old, l_old 这两个累积统计量？

```
（用自己话写）
```

## 2. V2 在 V1 基础上做了什么？

### 2.1 循环顺序对调

| | V1 | V2 |
|---|---|---|
| 外循环 | K, V | Q |
| 内循环 | Q | K, V |
| warps 之间是否要通信 | 是 | 否 |
| 哪些可以并行 | batch * heads | batch * heads * **seq_len** |

### 2.2 为什么 V2 更快？

- ___（自己填）

### 2.3 为什么 GPU 占用率更高？

- ___（自己填，提示：grid 多了一个维度）

## 3. Triton 实现的关键代码点

### 3.1 grid 划分

```python
grid = (cdiv(N, BLOCK_M), B * H, 1)
```

为什么是这个 grid？

### 3.2 m_i, l_i, acc 的更新顺序

```python
m_ij = max(qk, axis=1)
m_new = max(m_i, m_ij)
p = exp(qk - m_new[:, None])
alpha = exp(m_i - m_new)
l_new = alpha * l_i + sum(p, axis=1)
acc = alpha[:, None] * acc + p @ V  ← 注意先修正旧的 acc
m_i = m_new
l_i = l_new
```

为什么先修正 acc 再加新的？

### 3.3 最终归一化

```python
acc /= l_i[:, None]
```

为什么这一步是对的？

## 4. 我的 Benchmark 数据

```
（W8 跑完 benchmarks/bench_flashattn_vs_sdpa.py 后填）
```

## 5. 我踩过的坑

```
（W8 实现过程中踩的坑都写进来，这是博客最有价值的部分）
```
