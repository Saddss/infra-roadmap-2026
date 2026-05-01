# W23：多模态浅入门

## 目标（一周时间，不深挖）

能讲清三件事：

1. ViT 怎么把图像变成 token（patch embedding + position embedding）
2. Qwen2.5-VL 的 window attention 为什么能省算力
3. DiT 的扩散结构基本范式

**不要深挖**。面试官不会因为你不会 DiT 拒你，但能讲清基本范式可以加分。

## 必交产出物

- [ ] `notes/01-vit-basics.md` — ViT 基础
- [ ] `notes/02-qwen-vl.md` — Qwen2.5-VL window attention
- [ ] `notes/03-dit-basics.md` — DiT 基础范式
- [ ] `notes/04-multimodal-inference.md` — 多模态推理优化（vLLM v1 多模态）
- [ ] `vit-experiments/` 至少跑通一次 Qwen2.5-VL 推理

## 每天拆解（W23 一周）

- D1：ViT 论文 + 中文图解，写 notes/01
- D2：Qwen2.5-VL 技术报告精读，写 notes/02
- D3：跑通 Qwen2.5-VL 推理，给一张图让模型看
- D4：DiT 论文 + 知乎"DiT 详解"，写 notes/03
- D5：理解 vLLM v1 多模态调度（vision encoder + LLM 共享调度），写 notes/04

## 资料

- [博客园 · Qwen2.5-VL 技术报告精读](https://www.cnblogs.com/emergence/p/18873748)
- [Qwen2.5-VL 算法解析](https://jishuzhan.net/article/2045498484711817217)
- [《Qwen2.5-VL》论文精读笔记](https://www.wsisp.com/helps/55456.html)
- [InternVL-2.5 训练细节](https://devpress.csdn.net/v1/article/detail/144970962)
- 知乎搜 "DiT 详解"（at least 2 篇）
