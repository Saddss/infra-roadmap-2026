# ViT 基础

## 核心思想

把图像当成"序列"喂给 Transformer：

```
1. 图像切 patch（典型 14×14 像素）
2. 每个 patch flatten + 线性投影 → 一个 token
3. 加 [CLS] token（用于分类）和 position embedding
4. 标准 Transformer encoder
```

## Patch Embedding

```python
# 224x224 图像，patch_size=14
# → 16x16 = 256 个 patch
# → 256 个 token（再加 1 个 CLS = 257）
```

## 与 NLP Transformer 的区别

- 没有 causal mask（所有 patch 互相看）
- 用 2D position encoding（很多用 2D-RoPE，Qwen2.5-VL 是 2D-RoPE）
- CLS token 在最后做分类

## 计算复杂度问题

attention 复杂度 O(N²)，N = patch 数。当处理高分辨率图像（512×512 或更高）时：

```
patch_size=14, image=512x512 → ~1300 个 patch
attention: 1300² = 1.7M ops
```

→ 高分辨率原生输入很贵。

## 解决方案：window attention

把 N 个 patch 切成多个 window（典型 8×8 patch），window 内做 full attention，window 间不通信。

```
N=1300, window=64
→ ceil(1300/64) = 21 个 window
→ 每个 window 算 64² = 4096 ops
→ 总共 21 * 4096 ≈ 86K ops（vs 朴素 1.7M）
```

→ Qwen2.5-VL 用这招（window=112×112=64 patch）。

## Qwen2.5-VL 的设计

```
ViT 共 32 层
- 4 层 full self-attention
- 28 层 window attention（112×112）
```

**只在少数层做全局通信**，其他层用 window 省算力。

## 我的核心理解

```
（高分辨率原生输入 + window attention 是 Qwen2.5-VL 的两大杀招）
```

## 资料

- ViT 论文：An Image is Worth 16x16 Words
- [Qwen2.5-VL 技术报告精读](https://www.cnblogs.com/emergence/p/18873748)
