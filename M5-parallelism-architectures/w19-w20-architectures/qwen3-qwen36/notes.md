# Qwen 系演进 · Qwen3 → Qwen3-Next → Qwen3.6

## 时间线

| 版本 | 发布 | 关键创新 | 总参/激活 |
|---|---|---|---|
| Qwen3 | 2025 | MoE + Hybrid Thinking (`/think`/`/no_think`) | 235B / 22B |
| Qwen3-Next | 2025.9 | + Gated DeltaNet 混合 | 80B / 3B |
| **Qwen3.6-35B-A3B** | **2026.4.16** | + 多模态 + 思维保留 | **35B / 3B** |

## Qwen3 的 Hybrid Thinking（前置必修）

### 双模式融合

```
chat template 引入 /think /no_think 标志
默认进入思考模式，模型在 <think>...</think> 中输出推理链
用户可以指定不思考: /no_think → 直接给答案
```

### 思考预算（Thinking Budget）

```
当 <think> 长度达到阈值时，强制插入停止指令:
"Considering the limited time by the user, I have to give the solution based on the thinking directly now."
模型基于已积累的推理生成最终回复
```

### 训练流程

1. Stage 1: 长 CoT 冷启动 SFT
2. Stage 2: 推理 RL（AIME/LiveCodeBench 类奖励）
3. Stage 3: Thinking Mode Fusion（继续 SFT，融合 thinking 和 non-thinking）
4. Stage 4: 通用 RL

## Qwen3-Next（前置）

第一次在 Qwen 系列引入 Gated DeltaNet 混合架构：

```
3 层 Gated DeltaNet + 1 层 Gated Attention（3:1 比例）
```

为什么这么做？
- DeltaNet：线性注意力 O(L) 复杂度，但内容寻址精度差
- Gated Attention：全注意力（带门控），保留长程依赖精度
- 3:1 是经过消融实验得到的最优比例

## Qwen3.6-35B-A3B（2026.4.16，必考）

### 模型规格

```
总参: 35B / 激活: 3B
专家: 256 routed + 1 shared, 每 token 激活 8 路由
hidden_size: 2048
num_layers: 40
注意力: Gated DeltaNet (32 V头/16 QK头) + Gated Attention (16 Q头/2 KV头)
位置编码: RoPE 64 维
词表: 248,320
原生上下文: 262K, YaRN 扩展到 1M
```

### 层级结构（10 个 block 的 pattern）

```
每个 block = 3×(Gated DeltaNet → MoE) + 1×(Gated Attention → MoE)
共 40 层（10 个 block）
```

### Gated DeltaNet 核心

```
S_t = β_t ⊙ S_{t-1} + Δ_t ⊗ (K_t ⊗ V_t)

β_t (gate): 控制每步记忆保留比例
Δ_t (delta): 对特定记忆位置的精确更新
```

特点：
- O(1) 状态内存（不需要缓存全量 KV）
- O(L) 时间复杂度
- 能选择性"遗忘"

### 多模态

第一个原生多模态的 Qwen MoE：

- 集成视觉编码器
- 支持图像、视频、文档输入
- RealWorldQA 85.3, VideoMME 86.6

### 思维保留（Thinking Preservation）

新增：在多步骤 Agent 任务中保留完整推理上下文，避免重复思考。

## 部署

```bash
# vLLM ≥ 0.19.0 必须
pip install vllm>=0.19.0

vllm serve Qwen/Qwen3.6-35B-A3B \
  --port 8000 \
  --tensor-parallel-size 1 \  # 单 4090 24GB 也能跑 4-bit
  --max-model-len 32768 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

## 我的核心洞察

```
（Qwen 这条路线的一致逻辑：用线性注意力 O(L) 替代主流 attention，少量全注意力补精度）
（这条路线打的是 V4 的反面：V4 把 N 压成 N/m 再选 top-k；Qwen3.6 直接用 RNN-style 状态）
```

## 面试可讲

- Q1：Gated DeltaNet 和 Mamba2 有什么不同？
- Q2：3:1 比例怎么来的？
- Q3：Qwen3.6 和 V4 哪个适合长上下文？答案不绝对！
- Q4：思维保留怎么实现的（不需要 KV cache 的话）？

## 资料

- [博客园 · Qwen3.6-35B-A3B 全面评测](https://www.cnblogs.com/sing1ee/p/19885253)
- [Qwen3.5/3.6 混合注意力解析 · Gated DeltaNet + MoE 部署](https://dev.tekin.cn/blog/qwen3-5-hybrid-attention-gated-deltanet-moe-deployment)
- [Qwen 官方博客](https://qwen.ai/blog?id=qwen3.6-35b-a3b)
- [Qwen3 技术报告详解](https://blog.csdn.net/kebijuelun/article/details/148070438)
