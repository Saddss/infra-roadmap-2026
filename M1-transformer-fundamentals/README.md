# M1（W1-W4）：Transformer 与 Attention 全家桶

## 目标

能在白板上独立推导出 **MHA / MQA / GQA / MLA** 的 KV Cache 大小、计算量差异，并用 PyTorch 从零写一个能跑的 mini-Llama。

## 必交产出物

- [ ] `mini-llama/` 仓库（已搭好脚手架，你按周填代码）
- [ ] 一篇博客《我手撕 MHA→GQA→MLA 的笔记》（≥3000 字带图，发到知乎/掘金）

## 每周拆解

### W1：注意力数学原理 + 单头实现

- [ ] 看视频：B 站李宏毅《生成式 AI 导论》Transformer 章节
- [ ] 读：苏剑林《缓存与效果的极限拉扯》（[科学空间](https://kexue.fm/archives/10091)）
- [ ] 推导：Q/K/V 来源、scaled dot-product 公式、为什么除 √d
- [ ] 代码：填 `mini_llama/attention/single_head.py`（纯 numpy 单头注意力）
- [ ] 单测：`tests/test_single_head.py` 跑过

### W2：MHA → MQA → GQA 演进

- [ ] 读：[Yue Shui · MHA/MQA/GQA 对比](https://syhya.github.io/zh/posts/2025-01-16-group-query-attention/)
- [ ] 读：`meta-llama/llama` 的 `model.py` GQA 实现
- [ ] 代码：填 `mini_llama/attention/mha.py`、`mqa.py`、`gqa.py`
- [ ] 实验：在 `scripts/compare_kv_cache.py` 里输出三种注意力的 KV Cache 显存占用对比表

### W3：MLA + RoPE + 矩阵吸收

- [ ] 读：DeepSeek-V3 论文 Section 2.1（MLA + DeepSeekMoE）
- [ ] 读：[冷眸 · MLA 中矩阵吸收技巧](https://lengm.cn/post/20250226_attention/)
- [ ] 代码：填 `mini_llama/attention/mla.py`（latent vector + RoPE 解耦 + 矩阵吸收）
- [ ] 同时填：`mini_llama/positional/rope.py`（旋转位置编码）

### W4：组装一个能跑的 mini-Llama + 写博客

- [ ] 把 `nanoGPT` clone 下来对照学习
- [ ] 填 `mini_llama/model.py` 把 attention + MLP + RMSNorm + RoPE 组装成完整 LM
- [ ] 在 tinyshakespeare 数据集上跑通训练（小 corpus 即可）
- [ ] 切换 attention 实现（MHA / GQA / MLA），验证生成效果近似
- [ ] 写博客发出去

## 阶段切换检查表（进 M2 前）

- [ ] `mini-llama/` 已 push 到 GitHub，README 写清架构选型
- [ ] 博客已发布到知乎/掘金，**收到至少 3 个评论**才算通过（不是 3 个赞）
- [ ] `weekly-tracker.md` 4 周记录都填了
- [ ] 简历"项目"那一栏更新一行：mini-llama (Pytorch from scratch, supports MHA/GQA/MLA)

## 资料

→ 见 [`resources.md`](./resources.md)

## 周记

→ 见 [`weekly-tracker.md`](./weekly-tracker.md)
