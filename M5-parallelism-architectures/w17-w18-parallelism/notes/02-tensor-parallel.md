# 02 · 张量并行（TP）

## 核心思想

把单层的矩阵乘法切到多卡上：

- **行并行（Row-Parallel）**：把权重 W 按行切，输出需要 all-reduce
- **列并行（Column-Parallel）**：把权重 W 按列切，下一层做行并行就能省一次 all-reduce

## Megatron 的标准切法（Transformer block）

```
Attention:
  Q, K, V: 列并行（每张卡管一部分 head）
  Out projection: 行并行（all-reduce 在这里发生）

FFN:
  W1: 列并行
  W2: 行并行（all-reduce 在这里发生）
```

每个 transformer block 总共 4 次 all-reduce（attention 2 次 + FFN 2 次）。

## TP 的代价

- 每次前向反向都要 all-reduce
- 通信量大 → 必须放在 NVLink 内（同节点内）
- 跨节点 TP = 性能崩

## 与 SP（Sequence Parallel）配合

- SP 把 LayerNorm 和 dropout 也切到 sequence 维度
- 减少激活内存占用 ~1/TP 倍

## Megatron 启动参数

```bash
torchrun --nproc-per-node=4 pretrain_gpt.py \
  --tensor-model-parallel-size 4 \
  --sequence-parallel \   # 推荐和 TP 一起开
  ...
```

## 实验记录

```
4 卡 GPT-2-1.3B:
  TP=1, PP=4: __ samples/s
  TP=2, PP=2: __ samples/s
  TP=4, PP=1: __ samples/s（节点内同 NUMA）
```

## 我的发现

```
（什么场景下 TP 优？什么场景下 PP 优？TP 跨 NUMA 为什么会崩？）
```
