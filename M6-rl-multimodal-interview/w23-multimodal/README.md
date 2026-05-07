# W23：多模态浅入门

## 目标（一周时间）

能讲清这四件事：

1. ViT 怎么把图像变成 token（patch embedding + position embedding）
2. Qwen2.5-VL 的 window attention 为什么能省算力
3. DiT 的扩散结构基本范式
4. **omni-modal 推理系统**（vLLM-Omni / SGLang-Omni）解决什么 vLLM/SGLang 主项目解决不了的问题

**前 3 项不要深挖**。面试官不会因为你不会 DiT 拒你，但能讲清基本范式可以加分。
**第 4 项是 2026 上半年的新热点**，简历能写"了解 omni-modal 推理系统设计哲学"就够了。

## 必交产出物

- [ ] `notes/01-vit-basics.md` — ViT 基础
- [ ] `notes/02-qwen-vl.md` — Qwen2.5-VL window attention
- [ ] `notes/03-dit-basics.md` — DiT 基础范式
- [ ] `notes/04-multimodal-inference.md` — 多模态推理优化（vLLM v1 多模态）
- [ ] `notes/05-omni-modal-systems.md` — vLLM-Omni / SGLang-Omni 进阶专题（已提供完整大纲）
- [ ] `vit-experiments/` 至少跑通一次 Qwen2.5-VL 推理

## 每天拆解（W23 一周）

- D1：ViT 论文 + 中文图解，写 notes/01
- D2：Qwen2.5-VL 技术报告精读，写 notes/02
- D3：跑通 Qwen2.5-VL 推理，给一张图让模型看
- D4：DiT 论文 + 知乎"DiT 详解"，写 notes/03
- D5：理解 vLLM v1 多模态调度（vision encoder + LLM 共享调度），写 notes/04
- D6/D7（可选 / 周末）：**vLLM-Omni + SGLang-Omni 浅尝**，按 [`notes/05-omni-modal-systems.md`](./notes/05-omni-modal-systems.md) 的"阅读路径"前 3 篇看

## 资料

→ 见 [顶层 M6 resources.md](../resources.md) 的 W23 多模态部分
→ omni-modal 系统专题资料在 [`notes/05-omni-modal-systems.md`](./notes/05-omni-modal-systems.md)
