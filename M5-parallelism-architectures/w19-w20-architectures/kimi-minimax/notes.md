# Kimi K2 Thinking & MiniMax M2 · 2026 双线辨析

## Kimi K2 Thinking（2025.11）

### 架构

```
1T 总参 / 32B 激活
DeepSeek V3 架构 + Muon 优化器（Kimi 是 Muon 的发源地）
384 个 expert, 每 token 激活 8
上下文 128K → 256K（thinking 版）
```

### 特点

- 思维链超长（输出是输入的 3-5 倍）
- 端到端代理行为（思考 + 工具调用）
- 速度慢：34 tok/s

### 与 V4 的关系

- Kimi 提出 Muon → DeepSeek V4 用 Muon
- Kimi K2 沿用 V3 架构，没有 V4 的 CSA/HCA 创新
- DeepSeek V4 是 Kimi 路线的"加强版"

## Kimi Linear（2025.10）

### 架构

```
3 层 Kimi Delta Attention（线性） + 1 层全局 MLA（全注意力）
全局 MLA 用 NoPE（不带位置编码）
位置偏差由 Kimi Delta Attention 暗中消化
```

### 与 Qwen3.6 的差异

| | Kimi Linear | Qwen3.6 |
|---|---|---|
| 线性 attention | Kimi Delta Attention | Gated DeltaNet |
| 全注意力层 | 全局 MLA + NoPE | Gated Attention + RoPE |
| 比例 | 3:1 | 3:1 |
| MoE 数量 | 暂不公开 | 256 |

**两条路线极其相似**，都是 3:1 混合，区别在线性 attention 的具体实现（KDA 还是 Gated DeltaNet）。

## MiniMax M2（2025.11）

### 反向案例：弃用线性注意力

```
230B 总参 / 10B 激活
回归全注意力（M1 是 Lightning Attention 即线性注意力）
推理速度 93 tok/s（比 Kimi K2 Thinking 快 2.7×）
```

### 为什么反向？

MiniMax 团队的判断：

1. **线性注意力在长链推理时精度损失**：thinking 模式生成 3-5× tokens，每一步都靠之前的 KV，线性 attention 的精度损失会累积放大
2. **Benchmark 一致性问题**：线性 attention 在传统 benchmark（MMLU、AIME）上和全注意力差距不大，但在 Agent 长链任务（Tau2-bench）上差距拉开
3. **KV 命中率问题**：Agent 多轮场景的前缀命中率高，全注意力 + RadixAttention 直接吃 prefix sharing 红利

### 性能对比

| 维度 | Kimi K2 Thinking | MiniMax M2 |
|---|---|---|
| 总参 | 1T | 230B |
| 激活 | 32B | 10B |
| 速度 | 34 tok/s | **93 tok/s** |
| 多步规划（Tau2-bench） | 66.1% | **77.2%** |
| 工具链稳定性 | 稳定 | **更稳定** |

## 三条路线的总结表

| 路线 | 代表 | 核心 idea | 优势 | 劣势 |
|---|---|---|---|---|
| 压缩稀疏 | DeepSeek V4 | 压缩到 N/m，再 top-k 选 | 精度保持好、KV cache 极小 | 需要 indexer 训练，推理 kernel 复杂 |
| 门控线性 | Qwen3.6, Kimi Linear | RNN-style 状态 + 少量全注意力 | 状态大小 O(1)、超长上下文便宜 | 长链推理精度损失 |
| 全注意力 | MiniMax M2 | 不做奇技淫巧，靠 prefix caching | benchmark 一致性好、Agent 场景强 | 长上下文成本高 |

## 我的判断（写在博客里）

```
（哪条路线最终会赢？我自己怎么看？）

我的初步判断:
1. 短期（2026 上半）: V4 路线在 long context benchmark 上会领先，但 Qwen3.6 路线在端侧部署上吃香
2. 中期（2026 下半）: 三条路线会各自细分场景共存
3. 长期: 取决于 RL 训练能不能让线性 attention 学会"该忘的忘、该记的记"
```

## 资料

- [LLM 架构新趋势综述](https://devpress.csdn.net/v1/article/detail/154573876)
- [Kimi K2 Thinking vs MiniMax M2 全面对比](https://kimi-k2.org/zh/blog/17-kimi-k2-thinking-vs-minimax-m2)
- [开源推理模型巅峰对决](https://blog.gitcode.com/f044616c21b80a6f28db2249be988908.html)
