# M6 资源清单

> 收录原则见顶层 [RESOURCES.md](../RESOURCES.md) 的"收录与排序原则"。

## RL（W21-W22）

### 视觉化教学（**先看这个建立直觉**）

- ⭐⭐⭐ **changyeyu/LLM-RL-Visualized** — [GitHub 仓库](https://github.com/changyeyu/LLM-RL-Visualized)
  - 100+ 原创架构图覆盖 LLM/VLM/RL/PPO/GRPO/DPO/SFT/CoT 蒸馏
  - 作者出版了豆瓣高分书《大模型算法：强化学习、微调与对齐》
  - **权威性极高**，建议先翻一遍图谱再看具体算法

### R1 复现教程（**最佳入门实战**）

- ⭐⭐⭐ [Datawhale R1 中文复现教程](https://cloud.tencent.com/developer/article/2494040) — 3×A800 跑通 GRPO，约 420 元
- ⭐⭐ [HuggingFace · Mini-R1 复现 DeepSeek R1 灵光一现](https://hugging-face.cn/blog/open-r1/mini-r1-contdown-game)
- ⭐⭐ [Open-R1 · DeepSeek R1 复现项目代码详解](http://152.67.113.27/articles/OpenR1...)

### 算法解读（**含公式推导**）

- ⭐⭐⭐ **《大模型后训练中的 GRPO 算法剖析》** — [知乎 2025](https://zhuanlan.zhihu.com/p/2020218462119679502)
  - 从"为什么 LLM 后训练适合 GRPO"角度讲清楚 PPO → GRPO 演进
  - 对比表清晰（critic / advantage 来源 / 显存代价）
- ⭐⭐ [GRPO 算法与 PPO 算法的本质区别](https://www.x-techcon.com/article/22823.html) — 含调参经验和踩坑总结
- ⭐⭐ [柠檬 CC · RLHF 实战系列](https://www.limoncc.com/post/47f9ae3708f426c0/) — PPO/GRPO/REINFORCE++ 公式推导

### 框架

- ⭐⭐⭐ [OpenRLHF README 中文版](https://github.com/OpenRLHF/OpenRLHF/blob/main/README_zh.md) — Ray + vLLM + DeepSpeed 工业级 RL 框架
- 📦 字节 veRL · `volcengine/verl`
- 📦 HuggingFace TRL · `huggingface/trl`
- 📦 TinyZero（极简）· `Jiayi-Pan/TinyZero`

## 多模态（W23）

### Qwen2.5-VL

- ⭐⭐ [博客园 · Qwen2.5-VL 技术报告精读](https://www.cnblogs.com/emergence/p/18873748)
- ⭐⭐ [Qwen2.5-VL 算法解析](https://jishuzhan.net/article/2045498484711817217)
- ⭐⭐ [《Qwen2.5-VL》论文精读笔记](https://www.wsisp.com/helps/55456.html)

### InternVL

- ⭐⭐ [InternVL-2.5 训练细节](https://devpress.csdn.net/v1/article/detail/144970962)（动态分辨率切片）

### DiT

- 论文：Scalable Diffusion Models with Transformers
- 知乎搜 "DiT 详解"（建议至少 2 篇）
- Sora 技术报告

### Omni-Modal 推理系统（**2026 上半年新热点 · 简历加分项**）

> 完整阅读路线 + 面试 Q&A 见 [`w23-multimodal/notes/05-omni-modal-systems.md`](./w23-multimodal/notes/05-omni-modal-systems.md)

**vLLM-Omni（vllm-project 官方，4400+ star）**

- ⭐⭐⭐ **[arXiv:2602.02204 · vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models](https://arxiv.org/abs/2602.02204)** — 论文，**必读**，理解 Stage 抽象 + OmniConnector
- ⭐⭐⭐ [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni) — 官方仓库
- ⭐⭐⭐ [Disaggregated Inference for Omni-Modality Models 设计文档](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/disaggregated_inference/)
- ⭐⭐⭐ [Q1 2026 Roadmap RFC #1192 · Omni Connector 全分离架构](https://github.com/vllm-project/vllm-omni/issues/1192) — 看官方下半年规划，是研究选题灵感金矿
- ⭐⭐ [Qwen2.5-Omni 部署文档](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/examples/offline_inference/qwen2_5_omni/) — 含 prompt 格式 + 端到端代码
- ⭐⭐ [vLLM PR #26156 · Add support for audio in video](https://github.com/vllm-project/vllm/pull/26156) — 看官方怎么实现"video 里的 audio"
- ⭐⭐ Jimmy Song《用 vLLM-Omni 快速部署多模态推理》— [jimmysong.io 中文](https://jimmysong.io/zh/ai/vllm-omni/) — 中文部署快速上手

**SGLang-Omni（sgl-project 官方）**

- ⭐⭐⭐ [sgl-project/sglang-omni](https://github.com/sgl-project/sglang-omni) — 官方仓库
- ⭐⭐⭐ [RFC #16546 · SGLang-Omni Design](https://github.com/sgl-project/sglang/issues/16546) — 整体设计哲学（inter/intra disaggregation）
- ⭐⭐⭐ [Refactoring Proposal #188](https://github.com/sgl-project/sglang-omni/issues/188) — 官方对现有架构的反思（**真实 infra 演进案例**）
- ⭐⭐⭐ [Qwen3-Omni Refactor #92](https://github.com/sgl-project/sglang-omni/issues/92) — Qwen3-Omni Thinker 集成 + FusedMoE / VisionAttention 优化
- ⭐⭐ [Ming-Omni Support Design Spec #236](https://github.com/sgl-project/sglang-omni/issues/236) — 多 stage 管道实例
- ⭐⭐ [Qwen3-Omni Technical Report (arXiv:2509.17765)](https://arxiv.org/abs/2509.17765) — Thinker-Talker 架构论文，omni 模型必读

## 面试（W24）

### 八股库

- ⭐⭐⭐ [小林 coding · 530+ 大模型面试题](https://xiaolincoding.com/other/ai.html)（核心）
- ⭐⭐ [字节大模型一面真题](https://blog.csdn.net/2401_84033492/article/details/141093004)
- ⭐⭐ [字节大模型一二三面经](https://www.nowcoder.com/feed/main/detail/ef14567a048e4e3fa30cbe28eb5c59e3)
- ⭐⭐ [入职字节大模型岗面试分享](https://devpress.csdn.net/v1/article/detail/147386717)

### 算法

- LeetCode hot 100
- 牛客网"大模型算法岗"专区面经

### 内推

- 知乎搜 "大模型 infra 内推 2026"
- LinkedIn 找在 ByteDance / 阿里 / 月之暗面 的朋友
- 直接公众号联系：机器之心、量子位都有招聘版块

## 6 个月学完后必读的"系统性书"（可选）

- 《Designing Machine Learning Systems》（Chip Huyen）
- 张俊林《GPT 时代的大模型实战》
- 《大模型算法：强化学习、微调与对齐》（changyeyu，配 LLM-RL-Visualized 仓库）
