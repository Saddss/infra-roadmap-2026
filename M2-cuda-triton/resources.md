# M2 资源清单 · CUDA + Triton

> 收录原则见顶层 [RESOURCES.md](../RESOURCES.md) 的"收录与排序原则"。

## CUDA 入门主线

### 教材

- ⭐⭐⭐ **PMPP 第 4 版**（Programming Massively Parallel Processors，2022 出版）— 教材
- ⭐⭐⭐ **李理《PMPP 第四版》中文翻译** — [李理博客 / fancyerii.github.io 2024.2](https://fancyerii.github.io/2024/02/20/pmpp/) — **网上最完整的 PMPP 中文版**，覆盖 22 章。BBuf 自己 RESOURCES 里也收录
- ⭐⭐⭐ **smarterhhsu/PMPP-Learning** — [GitHub](https://github.com/smarterhhsu/PMPP-Learning) — 38 个 Exercise CUDA 实现 + 中文笔记
- ⭐⭐ Smarter《PMPP 导读》— [smarter.xin 2024](https://smarter.xin/posts/30730973/) — 学习路线 + 各章概览

### CUDA 实战 / 高阶资源（**圣地级**）

- ⭐⭐⭐ **BBuf [`how-to-optim-algorithm-in-cuda`](https://github.com/BBuf/how-to-optim-algorithm-in-cuda)** — [OneFlow 工程师维护] — 国内 CUDA/MLSys 学习圣地
- ⭐⭐⭐ **BBuf [`RESOURCES.md`](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/master/RESOURCES.md)** — **挂这个就够你挖一年**，国内 CUDA/Triton/MLSys 资源最全索引

### 视频

- ⭐⭐ CUDA-MODE 课程笔记系列（B 站搜 "CUDA-MODE 课程笔记"）— BBuf 等人翻译/讲解的 GPU MODE 课程
- ⭐⭐ 搜 "CUDA-MODE 课程笔记 第二课 PMPP 1-3 章"、"第四课 PMPP 4-5 章"、"第九课 归约"

### PMPP 重点章节

- ⭐⭐⭐ Chapter 5: Memory Architecture and Data Locality（**Tiling 核心**）
- ⭐⭐⭐ Chapter 10: Reduction（树形规约）
- ⭐⭐⭐ Chapter 11: Prefix Sum（Scan）
- ⭐⭐ Chapter 16: Deep Learning（卷积/MatMul 优化）

## Triton 入门主线

### 官方教程

- ⭐⭐⭐ [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/) — 必跑：
  - 01-vector-add（入门）
  - 02-fused-softmax（理解 reduction）
  - 03-matrix-multiplication（理解 tiling + autotune）
  - 06-fused-attention（FlashAttention V2）

### 中文教程

- ⭐⭐⭐ **BBuf《OpenAI Triton 入门笔记一/二/三》** — [微信公众号系列 2024](https://github.com/BBuf/how-to-optim-algorithm-in-cuda)
  - 笔记一：vector add / softmax 入门
  - 笔记二：LayerNorm / RMSNorm（含 FA2 仓库的 Triton 实现解析）
  - 笔记三：FusedAttention（FlashAttention V2 完整解读）

## FlashAttention 论文 + 中文图解（**M2 W8 主菜**）

### 中文图解（必读 · 国内最权威）

- ⭐⭐⭐ **猛猿《图解大模型计算加速系列：FlashAttention V1，从硬件到计算逻辑》** — [知乎 2023.11](https://zhuanlan.zhihu.com/p/669926191)
  - **BBuf 在自己 Triton 笔记里反复推荐**，中文圈 FA1 解读最权威
  - 按论文梳理 + 自己重新推导，含完整 forward/backward
- ⭐⭐⭐ **猛猿《图解大模型计算加速系列：Flash Attention V2，从原理到并行计算》** — [微信公众号 2023.7](https://mp.weixin.qq.com/s/5K6yNj23NmNLcAQofHcT4Q)
- ⭐⭐ 《FlashAttention 算法之美：极简推导版》— [微信公众号](https://mp.weixin.qq.com/s/hu5D1dmCFkeStxbXBE-czA)
  - 简化版，适合入门第一遍
  - 写明对标 Zihao Ye 的 "From Online Softmax to FlashAttention"

### 论文 + 英文权威解读

- ⭐⭐⭐ FlashAttention V1 论文（Tri Dao et al., 2022, arXiv:2205.14135）
- ⭐⭐⭐ FlashAttention V2 论文（Tri Dao, 2023, tridao.me）
- ⭐⭐⭐ FlashAttention V3 论文（2024，H100 优化）
- ⭐⭐ Zihao Ye《From Online Softmax to FlashAttention》（FlashInfer 作者写的完整推导）

## 你的环境（先确认）

- GPU: 你需要至少 1 张 NVIDIA GPU（4090 / A10 / A100 / 5090）才能跑 Triton/CUDA
- 没有 GPU？租 [autodl.com](https://www.autodl.com)：4090 约 1.5 元/小时，A100 约 6 元/小时
- 安装：`pip install triton`（需要 Linux + CUDA 11.6+）

## W8 博客模板

```markdown
# 我用 Triton 100 行写了一个 FlashAttention，并解释它为什么快

## 1. 标准 Attention 慢在哪里
   - 1.1 QK^T 矩阵 N×N 的显存爆炸
   - 1.2 频繁的 HBM 读写

## 2. FlashAttention 的两个核心技巧
   - 2.1 Tiling（把大矩阵切小块）
   - 2.2 Online Softmax（不缓存中间矩阵）

## 3. V2 在 V1 的基础上做了什么
   - 3.1 把 Q 移到外循环
   - 3.2 序列长度上并行

## 4. 我的 Triton 实现（逐段解读）
   - 4.1 grid 划分策略
   - 4.2 主循环里的 m_i / l_i 更新
   - 4.3 最终归一化

## 5. Benchmark：vs PyTorch SDPA / vs naive PyTorch
   （贴出对比图）

## 6. 我踩过的坑
```
