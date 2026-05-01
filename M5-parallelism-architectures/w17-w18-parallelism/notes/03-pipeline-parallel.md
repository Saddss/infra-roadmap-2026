# 03 · 流水线并行（PP）

## 核心思想

把模型按层切到多张卡。GPU0 算第 0-N/P 层，GPU1 算 N/P-2N/P 层...

## 调度方式演进

| 方法 | 关键创新 | bubble | 显存 |
|---|---|---|---|
| Naive PP | - | 大 | 大 |
| GPipe | 微批次 | 中 | 大 |
| PipeDream / 1F1B | 提前反向 | 小 | 中 |
| PipeDream-2BW | 双缓冲权重 | 小 | 小 |
| Megatron Interleaved | 每卡多个 chunk | 极小 | 中 |

## 1F1B（关键）

```
GPU0: F1 F2 F3 F4 B1 F5 B2 F6 B3 ...
GPU1:    F1 F2 F3 F4 B1 F5 B2 ...
GPU2:       F1 F2 F3 F4 B1 ...
GPU3:          F1 F2 F3 F4 ...
```

stage 越后开始的越晚（startup phase），稳定后 forward / backward 交替。

## Bubble 公式

```
bubble = (P - 1) / M
P: pipeline stages
M: micro-batches per global batch
```

→ 增大 micro-batch 数能减小 bubble，但有上限（显存约束）

## Interleaved 1F1B（Megatron 推荐）

每卡 v 个 model chunk → bubble 缩小到 (P-1)/(M*v)

代价：通信次数变多。

## Megatron 启动参数

```bash
--pipeline-model-parallel-size 4 \
--virtual-pipeline-model-parallel-size 2 \  # interleaved
--num-layers-per-virtual-pipeline-stage 4
```

## 实验记录

```
GPT-2-1.3B, 4 卡:
  micro_batch=1:  bubble __ %
  micro_batch=4:  bubble __ %
  interleaved v=2: bubble __ %
```

## 我的发现

```
（为什么 Megatron 推荐 PP 跨节点 + TP 节点内？）
```
