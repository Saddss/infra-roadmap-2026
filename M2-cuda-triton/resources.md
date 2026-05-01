# M2 资源清单 · CUDA + Triton

## CUDA 入门主线

### 教材

- 📚 **Programming Massively Parallel Processors (PMPP) 第 4 版**（必读，2022 出版）
- 📦 中文笔记 + 38 Exercise CUDA 实现：[smarterhhsu/PMPP-Learning](https://github.com/smarterhhsu/PMPP-Learning)
- 📝 [PMPP 导读 · Smarter's blog](https://smarter.xin/posts/30730973/)

### 视频（B 站）

- 搜 "CUDA-MODE 课程笔记 BBuf"（必看，全中文）
- 搜 "GPU 编程入门 李硕"

### 实战参考

- 📦 [BBuf/how-to-optim-algorithm-in-cuda](https://github.com/BBuf/how-to-optim-algorithm-in-cuda)（国内 CUDA 学习圣地）
- 📝 [BBuf · CUDA 笔记系列十三：OpenAI Triton 入门笔记一](https://mp.weixin.qq.com/s/y0fYn8gUqHqEoRO41ftKnA)

### PMPP 重点章节

- ⭐ Chapter 5: Memory Architecture and Data Locality（Tiling 核心）
- ⭐ Chapter 10: Reduction（树形规约）
- ⭐ Chapter 11: Prefix Sum（Scan）
- Chapter 16: Deep Learning（卷积/MatMul 优化）

## Triton 入门主线

### 官方教程

- [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/)
  - 01-vector-add
  - 02-fused-softmax
  - 03-matrix-multiplication
  - 04-low-memory-dropout
  - 05-layer-norm
  - 06-fused-attention（FlashAttention V2）

### 中文教程

- 📝 [BBuf · OpenAI Triton 入门笔记三 FusedAttention](https://cloud.tencent.com/developer/article/2392140)
- 📝 [AI-HPC 联盟 · Triton 学习路径](https://ai-hpc.org/guide/08-compiler/triton-learning-path)
- 📝 [SegmentFault · 使用 Triton 实现 FlashAttention2](https://segmentfault.com/a/1190000047406258)

## FlashAttention 论文 + 中文图解

### 论文

- FlashAttention V1: Tri Dao et al., 2022
- FlashAttention V2: Tri Dao, 2023
- FlashAttention V3: 2024（H100 优化）

### 中文图解（必读）

- 📝 [图解大模型计算加速系列：FlashAttention V1，从硬件到计算逻辑](https://zhuanlan.zhihu.com/p/669926191)
- 📝 [图解大模型计算加速系列：Flash Attention V2，从原理到并行计算](https://mp.weixin.qq.com/s/5K6yNj23NmNLcAQofHcT4Q)

## 你的环境（先确认）

- GPU: 你需要至少 1 张 NVIDIA GPU（4090 / A10 / A100）才能跑 Triton/CUDA
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
