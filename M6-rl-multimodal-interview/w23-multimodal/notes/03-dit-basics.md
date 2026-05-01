# DiT 基础（Diffusion Transformer）

> 浅入门。能讲清基本范式即可。

## 核心思想

把扩散模型的 U-Net backbone 换成 Transformer。Sora 用的就是 DiT。

## 经典扩散模型回顾（30 秒版本）

```
Forward: 给图像逐步加噪声 x_0 → x_T（接近纯噪声）
Reverse: 学一个网络从 x_t 预测 x_{t-1}（去噪）
推理: 从噪声 x_T 一步步去噪到 x_0
```

去噪网络的输入：(x_t, t) → 预测噪声 ε

## DiT 怎么做

```
1. 把图像 patchify（同 ViT，typical 2×2 patch on latent space）
2. 加 timestep embedding 和 class embedding
3. 多层 DiT Block（带 adaLN 注入条件）
4. 最后一层投影回噪声预测
```

## DiT Block 结构

```
input → adaLN(scale, shift, timestep) → MHA
     → adaLN(scale, shift, timestep) → MLP
     → output
```

adaLN（adaptive Layer Norm）：用 timestep + condition 调制 LN 的 scale/shift。

## 为什么用 Transformer

- 比 U-Net 容易扩 scale（训 13B 的 U-Net 很难，DiT 容易）
- 注意力机制对长距离结构更好（Sora 的 minute 级视频靠这个）
- 视频可以当作 "spatiotemporal patches" 序列

## 推理优化

- 多步采样（典型 50-100 步）→ 总计算量是普通 forward 的 50-100 倍
- 加速方法：DPM-Solver、Flash Attention、FP8

## 与 LLM 推理的关键差异

| | LLM | DiT |
|---|---|---|
| 自回归 | 是 | 否 |
| KV Cache | 有 | 无（每步独立） |
| 多次 forward | 1 次 | 50-100 次 |
| 主要开销 | memory-bound | compute-bound |

## 我的核心理解

```
（DiT 是把"图像生成"问题用 Transformer 解，但优化思路和 LLM 推理完全不同）
```

## 面试可讲

- Q1：DiT vs U-Net 区别？
- Q2：DiT 推理怎么加速？
- Q3：Sora 怎么处理视频的时间维度？
- Q4：扩散模型推理能用 PagedAttention 吗？

## 资料

- DiT 论文：Scalable Diffusion Models with Transformers
- 知乎搜 "DiT 详解"
- Sora 技术报告
