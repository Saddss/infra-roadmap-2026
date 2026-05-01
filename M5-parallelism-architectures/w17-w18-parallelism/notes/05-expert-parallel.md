# 05 · 专家并行（EP）—— 给 MoE

## 核心思想

把不同 expert 放到不同卡上：

- 每个 token 经过 router 后路由到 top-k expert
- 路由结果做 all-to-all 通信，把 token 发到对应卡
- expert 计算完后 all-to-all 发回原卡

## 与传统 DP 的区别

```
DP:    每卡持有完整模型副本
EP:    每卡持有不同的 expert 子集 → 卡少 expert 多时省显存
EP+TP: expert 内部再做 TP（V3-V4 用过）
```

## DeepSeek-V3 的 EP 配置

```
1024 GPU:
  TP=2, PP=16, EP=64
  每卡持有 256 / 64 = 4 个 expert
```

## DeepSeek-V4 的变化

V4 把 routed expert 数从 256 → 384，shared expert 仍为 1：

```
V4-Pro:  EP=64,  每卡 6 个 routed expert
V4-Flash: EP=32, 每卡 8 个 routed expert
```

## Megatron 启动参数

```bash
--expert-model-parallel-size 8 \
--num-experts 64 \
--moe-router-topk 2 \
--sequence-parallel  # 与 EP 联用必须开 SP
```

## 关键约束

**EP + TP 必须开 SP**：因为 EP 的 all-to-all 切的是 sequence 维度，要和 TP 的激活切分对齐。

## 通信开销

- Top-2 routing → 每个 token 在每层做 2 次 all-to-all（forward 1 + backward 1）
- all-to-all 比 all-reduce 贵 2-3×
- 必须在节点内 / 高带宽网络上做

## 我的发现

```
（Routed expert 数和 GPU 数怎么搭配？load balancing loss 重要吗？）
```
