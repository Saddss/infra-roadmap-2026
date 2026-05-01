# M1 资源清单 · 注意力变体

## 主线阅读（按推荐顺序）

1. ⭐ **苏剑林《缓存与效果的极限拉扯：从 MHA、MQA、GQA 到 MLA》** — [科学空间](https://kexue.fm/archives/10091)
   - 必读。中文最权威的 attention 演进综述
2. ⭐ **Yue Shui《Transformer 注意力机制：MHA、MQA 与 GQA 的对比》** — [syhya.github.io](https://syhya.github.io/zh/posts/2025-01-16-group-query-attention/)
   - 配公式 + 图，适合跟着推导
3. **冷眸《Attention 各种变体：全面详解 MHA/MQA/GQA/MLA》** — [lengm.cn](https://lengm.cn/post/20250226_attention/)
   - MLA 部分讲得最细，含矩阵吸收技巧
4. **腾讯云《MHA/MQA/GQA/MLA 注意力机制对比分析》** — [cloud.tencent.com](https://cloud.tencent.com/developer/article/2634499)
   - 业界视角，解释为什么各家选不同方案

## 视频（B 站）

- 李宏毅《生成式 AI 导论》Transformer 章节
- 李沐《动手学深度学习》Transformer 章节（自带字幕）
- 王树森《Transformer 模型》系列（最经典）
- 搜 "MLA DeepSeek 详解"、"RoPE 旋转位置编码 详解"

## 论文（按必读优先级）

| 论文 | 出处 | 必读级 |
|---|---|---|
| Attention Is All You Need (Transformer) | NeurIPS 2017 | ⭐⭐⭐ |
| Fast Transformer Decoding (MQA) | 2019 | ⭐⭐ |
| GQA: Training Generalized Multi-Query Transformer | 2023 | ⭐⭐⭐ |
| DeepSeek-V2 / V3 Technical Report (MLA) | 2024 | ⭐⭐⭐ |
| RoFormer (RoPE) | 2021 | ⭐⭐⭐ |
| Root Mean Square Layer Normalization (RMSNorm) | 2019 | ⭐⭐ |

## 源码仓库（顺序读）

```
1. karpathy/nanoGPT          # 500 行 GPT-2 复现，必读
2. meta-llama/llama           # llama2 model.py，看 GQA + RoPE 实战
3. deepseek-ai/DeepSeek-V3    # model.py 中 MLA 实现
```

## W4 博客模板（按这个结构写）

```markdown
# 我手撕 MHA→GQA→MLA 的笔记 —— 一个推理工程师的注意力机制重学

## 1. 起点：经典 MHA

### 1.1 数学公式
### 1.2 我的 PyTorch 实现（< 30 行）
### 1.3 KV Cache 显存占用计算

## 2. MQA：极致压缩 KV

### 2.1 共享 K/V 头的核心思想
### 2.2 代价：模型质量下降

## 3. GQA：折中方案

### 3.1 分组共享
### 3.2 为什么 LLaMA-2/3 选 g=8

## 4. MLA：DeepSeek 的杀手锏

### 4.1 低秩 latent vector
### 4.2 矩阵吸收技巧（推理时计算量不增）
### 4.3 RoPE 怎么和 MLA 兼容（解耦设计）

## 5. 一张表对比四种方案

## 6. 我跑出来的 KV Cache 显存对比实验
```
