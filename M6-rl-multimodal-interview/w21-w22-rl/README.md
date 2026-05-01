# W21-W22：RL 训推一体

## 目标

跑通 R1 类复现（GRPO + Qwen2.5-1.5B + Countdown 任务），理解 vLLM 在 RL 训练中作为 rollout engine 的角色。

## 必交产出物

- [ ] `r1-reproduction/` 中跑通 GRPO 训练
- [ ] `notes/` 中 4 份算法笔记
- [ ] 一段博客《我跑通 R1 复现的踩坑记录》

## 每周拆解

### W21：算法理论 + 框架对比

- [ ] 看 [Datawhale R1 中文复现教程](https://cloud.tencent.com/developer/article/2494040)
- [ ] 看 [HuggingFace Mini-R1 教程](https://hugging-face.cn/blog/open-r1/mini-r1-contdown-game)
- [ ] 写 4 份笔记：
  - [ ] `notes/01-ppo.md` — PPO 基础（带 critic）
  - [ ] `notes/02-dpo.md` — DPO（无 critic、无 RM、直接用偏好对）
  - [ ] `notes/03-grpo.md` — GRPO（DeepSeekMath 提的，无 critic、用 group 平均做 baseline）
  - [ ] `notes/04-reinforce-plus-plus.md` — REINFORCE++（OpenRLHF 力推，无 critic、按 token KL）

### W22：跑通复现 + 写博客

- [ ] 用 OpenRLHF 或 TRL（HF）跑通 GRPO 训练
- [ ] 模型：Qwen2.5-1.5B-Instruct（小但够练手）
- [ ] 任务：Countdown game（数学推理）
- [ ] 硬件：3×A800 80GB（云租约 420 元）
- [ ] 训完之后保存模型 + 测试推理质量
- [ ] 写博客《我跑通 R1 复现的踩坑记录》

## 资料

- [Datawhale R1 中文复现教程](https://cloud.tencent.com/developer/article/2494040)
- [HuggingFace Mini-R1 复现 DeepSeek R1](https://hugging-face.cn/blog/open-r1/mini-r1-contdown-game)
- [Open-R1 · 复现项目代码详解](http://152.67.113.27/articles/OpenR1...)
- [OpenRLHF README 中文版](https://github.com/OpenRLHF/OpenRLHF/blob/main/README_zh.md)
- [柠檬 CC · RLHF 实战系列](https://www.limoncc.com/post/47f9ae3708f426c0/)

## 框架选择建议

| 框架 | 优势 | 适合 |
|---|---|---|
| TRL（HF） | 易上手，文档全 | 教学 / 简单实验 |
| OpenRLHF | Ray + vLLM 分布式，工业级 | 大规模训练 |
| veRL（字节） | 字节自家，性能好 | 大公司风格 |
| TinyZero | 极简实现 | 学习用 |

W21-W22 推荐用 **OpenRLHF**（最贴近大厂工业用法 + 强对接 vLLM）。
