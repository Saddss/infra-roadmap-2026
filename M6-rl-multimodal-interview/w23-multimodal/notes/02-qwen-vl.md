# Qwen2.5-VL · 多模态推理代表作

## 整体架构

```
Image / Video → ViT (32 层, 大部分用 window attention)
            → MLP Vision-Language Merger（4 个 patch → 1 个 LLM token）
            → LLM (Qwen2.5)
            → 输出
```

三大组件：

1. **LLM**：Qwen2.5 LLM 作为基础（GQA + RoPE + RMSNorm + SwiGLU）
2. **Vision Encoder**：重新设计的 ViT
3. **MLP Vision-Language Merger**：把视觉 token 压缩送给 LLM

## ViT 配置（3B/7B/72B 统一）

```
hidden_size: 1280
layers: 32
heads: 16
patch_size: 14
window_size: 112  (= 8x8 patches)
全局 attention 层: 第 7, 15, 23, 31 层（仅 4 层）
其他层: window attention
归一化: RMSNorm
激活: SwiGLU
位置编码: 2D-RoPE
```

## 视觉 Token 压缩（Vision-Language Merger）

```
ViT 输出 patch token: 比如 1300 个
Merger: 把空间相邻的 4 个 patch 拼接 → 1 个 LLM token
        2 层 MLP 投影到 LLM hidden_size
→ 1300 / 4 = 325 个 LLM token
```

**为什么需要**：高分辨率视觉 token 数量太多，会挤占 LLM context。Merger 把它们压成 1/4。

## MRoPE（多模态 RoPE）

把位置编码拆成三个分量：

```
position_id = (time_id, height_id, width_id)
```

- 文本输入：三个 ID 相同（退化为 1D RoPE）
- 图像：根据 (h, w) 分配
- 视频：time_id 对齐到**真实时间**（不是帧序号），所以不同 fps 的视频能学到一致语义

## 视频处理

```
14×14 patch 作为图像基本单元
2 个连续帧组合 → 减少视觉 token 数
动态 FPS 采样
绝对时间编码（MRoPE 的关键改进）
```

## 训练效率优化

视觉 token 数随分辨率变化 → 不同样本计算量差异大 → GPU 负载不均衡

解法：根据输入到 LLM 的总 token 数动态打包样本（dynamic packing）。

## 推理优化（W23 重点）

vLLM v1 多模态新特性：

```
vision encoder + LLM decoder 共享调度（v1 vs v0 主要差异）
图像 token 也可以做 prefix caching（多张相同图）
```

## 我的核心理解

```
（Qwen2.5-VL 的"三个核心"：
  1. window attention 让 ViT 复杂度从 O(N²) 降到 O(N*window²)
  2. MLP Merger 把视觉 token 压成 1/4
  3. MRoPE 用绝对时间对齐，跨 FPS 学一致语义
这三招让"原生分辨率 + 多模态"变得可工程化）
```

## 面试可讲

- Q1：Qwen2.5-VL 的 ViT 为什么不用 224×224 固定输入？
- Q2：window attention 为什么有用？什么场景失效？
- Q3：MRoPE 和 1D RoPE 区别？
- Q4：多模态推理的 KV cache 怎么管？
