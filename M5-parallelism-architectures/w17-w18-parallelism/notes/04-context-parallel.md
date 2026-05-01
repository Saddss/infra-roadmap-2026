# 04 · 上下文并行（CP）

## 核心思想

把序列长度切分到多张卡。每卡处理 `seq_len / CP` 个 token。

适用场景：长上下文训练（>= 32K）。

## 实现要点

- 每卡持有一段 token 的 Q/K/V
- attention 计算时需要 ring 通信：每卡的 K/V 轮流发给其他卡
- 通常和 FlashAttention 配合（环形 KV 通信）

## Megatron 启动参数

```bash
--context-parallel-size 2
```

## 与 SP 的区别

| | SP | CP |
|---|---|---|
| 切什么 | LayerNorm / dropout 输入 | attention 的 Q/K/V |
| 通信 | all-gather + reduce-scatter | ring p2p |
| 节省什么 | 激活内存 | 计算 + 内存 |
| 与 TP 关系 | 必须配合 | 独立 |

## 与 EP 的关系

EP 切 expert，CP 切 sequence，**正交**。70B+ MoE 模型经常 EP × CP 同时开。

## 我的发现

```
（什么时候必须开 CP？开了之后 throughput 下降多少？）
```
