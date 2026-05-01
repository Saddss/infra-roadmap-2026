# 06 · 序列并行（SP）

## 核心思想

在 TP 的基础上，把 LayerNorm / Dropout / RMSNorm 这些原本"在所有 token 上做相同操作"的层也切到 sequence 维度。

## 为什么需要 SP

TP 切了矩阵乘法，但 LayerNorm 这种 element-wise 操作在每张卡上重复算 → 激活内存浪费。

## 怎么切

```
原来 (TP only):
  forward: all-reduce (after FFN out_proj)
  LayerNorm: 全部 token 都在每张卡上算 → 激活内存爆炸

加了 SP:
  forward: reduce-scatter (after FFN out_proj)  ← 切 sequence 后散
  LayerNorm: 每卡只算自己那段 sequence
  下次进 attention 前: all-gather 把 sequence 取齐
```

## 通信 trade-off

- TP-only: 每个 block 4 次 all-reduce
- TP + SP: 每个 block 4 次 reduce-scatter + 4 次 all-gather

reduce-scatter + all-gather ≈ all-reduce（NCCL 内部就是这么实现的），所以**通信总量不变，但激活内存 ↓ TP 倍**。

## Megatron 启动参数

```bash
--tensor-model-parallel-size 4 \
--sequence-parallel
```

## 我的发现

```
（为什么 SP 在 7B 以下模型收益不明显？）
```
