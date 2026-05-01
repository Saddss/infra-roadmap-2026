# DeepSeek 系演进 · V3 → V3.2 → V4

## 时间线

| 版本 | 发布日期 | 关键创新 | 总参/激活 |
|---|---|---|---|
| V3 | 2024.12 | MLA + DeepSeekMoE + FP8 训练 | 671B / 37B |
| V3.2-Exp | 2025.9 | + DSA（Lightning Indexer + Top-K 稀疏） | 671B / 37B |
| **V4-Pro** | **2026.4.24** | + mHC + CSA/HCA + Muon | **1.6T / 49B** |
| V4-Flash | 2026.4.24 | 同上，体量减小 | 284B / 13B |

## V3 的 MLA（前置必修）

### 核心思想

把输入压成低维 latent vector c（维度 512），KV Cache 只缓存 c。

```
传统 MHA cache: 2 * num_heads * head_dim = 2 * 128 * 128 = 14336 维 / token
MLA cache:      d_c + d_rope = 512 + 64 = 576 维 / token
→ MLA = MHA 的 1/25
```

### RoPE 解耦

MLA 不能直接和 RoPE 兼容（RoPE 不是线性变换，无法和矩阵吸收技巧合并），所以：

- KV 分两部分：nope 部分（不带 RoPE，维度 512）+ rope 部分（带 RoPE，维度 64）
- nope 部分用 latent vector c 压缩
- rope 部分独立缓存

## V3.2-Exp 的 DSA（前置必修）

### Lightning Indexer

对每个 query 计算"分数"，选出 top-k=2048 最相关的 KV：

```python
# DSA Lightning Indexer 核心公式（论文 Equation 1）:
# I_{t,s} = sum_j w^I_{t,j} * ReLU(q^I_{t,j} . k^I_s)
```

**关键属性**：
- 用 FP8 计算（更快）
- 每个 query 都得到 top-k 索引
- 这个索引在 prefill 和 decode 阶段都有用

### 训练阶段

1. **Dense warm-up**：只训 indexer，对齐原始 attention 分布
2. **Sparse training**：引入 top-k 选择，端到端训练

## V4 的三大创新（2026 必考）

### 1. mHC（Manifold-Constrained Hyper-Connections）

```
传统残差: x → x + f(x)
mHC: x → x_mixed = B @ x_pre, 其中 B 约束在 Birkhoff 多面体（双随机矩阵）

Birkhoff 多面体: 行和=1, 列和=1
→ 谱范数 ≤ 1 → 残差变换为非扩展映射 → 训练稳定
```

实现细节：
- 用 Sinkhorn-Knopp 20 次迭代把 B 投影到 Birkhoff 流形
- A_l 和 C_l 用 Sigmoid 约束为非负有界
- 三个映射都是静态参数 + 动态参数 组合

### 2. CSA + HCA 混合注意力（替代 MLA）

#### CSA（Compressed Sparse Attention）

```
Step 1: KV 压缩 — 每 m=4 个 token 压成 1 个 KV entry
Step 2: Lightning Indexer + top-k 选择
Step 3: 在 top-k 个压缩 KV 上做 MQA
Step 4: Grouped output projection（head_dim=512，分 g 组）
```

→ 序列长度 N 降到 N/m，再降到 top-k → **两层压缩**

#### HCA（Heavily Compressed Attention）

```
更激进：每 m'=128 个 token 压 1 个 KV
不做稀疏选择，全量密集 attention
```

#### CSA / HCA 交替叠加

```
V4-Pro:  61 层
V4-Flash: 43 层
```

CSA 和 HCA 一层一层往上叠。既不漏细节，也不被细节拖住。

#### 辅助设计

- **Q/KV RMSNorm**：core attention 前对 query/KV 做归一化（防 logit 爆炸）
- **Partial RoPE**：仅对最后 64 维施加 RoPE
- **滑动窗口**：附加一个小窗口（n_win=128）的未压缩 KV
- **Attention Sink**：可学习的 sink logit，允许总分 ≠ 1

#### 1M 上下文 KV Cache 仅为 GQA8 基线的 **2%**

### 3. Muon 优化器

替代 AdamW，hybrid Newton-Schulz 10 步迭代。

**首个在大规模 MoE 上成功应用 Muon 的工作**。

## V4 详细参数对比

| 参数 | V4-Flash | V4-Pro |
|---|---|---|
| Transformer 层数 | 43 | 61 |
| 隐藏维度 | 4096 | 7168 |
| CSA 压缩率 m | 4 | 4 |
| HCA 压缩率 m' | 128 | 128 |
| 查询头数 | 64 | 128 |
| 头维度 | 512 | 512 |
| 路由专家 | 256 | 384 |
| 共享专家 | 1 | 1 |
| 每 token 激活专家 | 6 | 6 |
| SWA 窗口 | 128 | 128 |
| 总参数 | 284B | 1.6T |
| 激活参数 | 13B | 49B |
| 训练 tokens | 32T | 33T |

## 我的核心洞察

```
（DeepSeek 这条路线的一致逻辑：把全局 attention 用各种"压缩 + 稀疏"方法降到次平方）
```

## 面试可讲

- Q1：V3.2 的 DSA 和 V4 的 CSA 区别？
- Q2：Lightning Indexer 为什么用 FP8？
- Q3：V4 用 GQA8 做基线对比，为什么不和 MLA 比？
- Q4：mHC 解决了 V3 的什么问题？

## 资料

- [量子位 · V4 报告太详尽了](https://www.163.com/dy/article/KRBULUJ60511DSSR.html)
- [CSDN · V4 技术报告全解读](https://deepseek.csdn.net/69f040bc0a2f6a37c5a685c2.html)
- [掘金 · V4 简要解读 · 含详细参数对比表](https://juejin.cn/post/7631898635937497134)
- [机器之心 · DSA 公开解读](https://mp.weixin.qq.com/s/WYze9rEZnuZ9l1Y132VJmA)
- [CSDN · DSA 算法源码分析](https://deepseek.csdn.net/69f0799e54b52172bc7087e5.html)
