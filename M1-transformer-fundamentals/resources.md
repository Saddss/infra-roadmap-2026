# M1 资源清单 · 注意力变体

> 收录原则见顶层 [RESOURCES.md](../RESOURCES.md) 的"收录与排序原则"。
> 每条 `[作者背景, 平台 YYYY.MM]` 让你判断权威度和时效。

## 主线阅读（按推荐顺序）

### 苏剑林系列（**两篇都要读，必须按顺序**）

1. ⭐⭐⭐ **苏剑林《缓存与效果的极限拉扯：从 MHA、MQA、GQA 到 MLA》** — [追一科技算法研究员 / kexue.fm 2024.5](https://kexue.fm/archives/10091)
   - 中文圈 attention 演进最权威综述
   - 配套理解：MLA 怎么"用低秩投影 + 恒等变换 + RoPE 解耦"压缩 KV Cache

2. ⭐⭐⭐ **苏剑林《Transformer 升级之路：20、MLA 好在哪里？(上)》** — [kexue.fm 2025.5](https://kexue.fm/archives/10907)
   - **必读续作**！作者本人和 DeepSeek 团队讨论过 MLA
   - 关键新论点："MLA 本质是 NoPE 的 MHA，训练时 head_dims=192，decode 时变 MQA-512"
   - 颠覆了"MLA 只是简单的低秩压缩"的浅层理解
   - 评论区有 zhihu 同行追问 + 作者本人 5+ 条详细回复，含金量极高

### zartbot 系列（**国内一线推理工程师，DeepSeek 团队认证过**）

3. ⭐⭐ **zartbot《从 MHA 到 MLA 看 Attention 优化：谈谈 DeepSeek 拼多多级的推理价格》** — [扎波特的橡皮擦 / 微信公众号 2024.5](https://mp.weixin.qq.com/s/l2uUXGQ-8Rj_nI3JG5lZ_g)
   - 从代数视角看 MLA 的低秩分解
   - 解释为什么"训练阶段拆开、推理阶段合并"是优雅设计

4. ⭐⭐ **zartbot《继续谈谈 MLA 以及 DeepSeek-MoE 和 SnowFlake Dense-MoE》** — [微信公众号 2024.5](https://mp.weixin.qq.com/s/hI7q4_-ZMtFIQ-ckhM9_YQ)
   - 上一篇姊妹篇，参数量分析（**MLA 的 KV 参数量仅为 MHA 的 11%**）
   - 配 SnowFlake Dense-MoE 对比

5. ⭐⭐ **zartbot《详细谈谈 DeepSeek MoE 相关的技术发展》** — [微信公众号 2025.2](https://mp.weixin.qq.com/s/WFJxnTF9fGIIXPA7GQ5V2w)
   - 文中明确说"得到了 DeepSeek 团队同学的指正"，权威性强
   - V2 → V3 MoE 演进，专家分组、Sigmoid 替代 Softmax

### 综述类

6. ⭐⭐ **Yue Shui《Transformer 注意力机制：MHA、MQA 与 GQA 的对比》** — [syhya.github.io 2025.1](https://syhya.github.io/zh/posts/2025-01-16-group-query-attention/)
   - 配公式 + 图，适合跟着推导

## 视频（B 站）

- 李宏毅《生成式 AI 导论》Transformer 章节（中文，自带字幕）
- 李沐《动手学深度学习》Transformer 章节
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
2. meta-llama/llama          # llama2 model.py，看 GQA + RoPE 实战
3. deepseek-ai/DeepSeek-V3   # model.py 中 MLA 实现
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
### 4.4 苏剑林的新理解："MLA 本质是 NoPE 的 MHA"
   ← 这一节可以让博客比 90% 的同类文高一档

## 5. 一张表对比四种方案

## 6. 我跑出来的 KV Cache 显存对比实验
```
