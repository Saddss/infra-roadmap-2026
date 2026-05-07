# 05 · Omni-Modal 推理系统：vLLM-Omni 和 SGLang-Omni

> W23 进阶专题。多模态 infra 的"下一站"是从"VLM"（视觉+语言）走向"omni-modal"（任意模态进任意模态出）。
> vLLM 和 SGLang 都已经为此各自发布了**官方独立子项目**，这是 2026 上半年最值得跟踪的两个 infra 工程。

> 收录原则见顶层 [RESOURCES.md](../../../RESOURCES.md) 的"收录与排序原则"。

## 0. 为什么"omni-modal"是个新问题

经典 VLM（Qwen2.5-VL / InternVL）：

```
图像 / 视频 → ViT encoder → LLM 解码 → 文本输出
                                   ↑ 单输出模态
```

omni 模型（Qwen2.5-Omni / Qwen3-Omni / Ming-Omni / Fish Audio S2 Pro）：

```
文本 + 图像 + 视频 + 音频  →  Thinker (LLM) → Talker (语音生成)
       ↑ 任意输入组合                          ↓ 多输出模态（文本 + 语音 + ...）
```

**新挑战**：
- 多个 encoder（vision / audio）和多个 decoder（LLM / TTS / DiT）需要协同调度
- AR（自回归 LLM）和非 AR（DiT 扩散）混合在同一推理图
- 不同 stage 的资源需求差异巨大（encoder 一次性、decoder 流式）
- 现有 vLLM/SGLang 的 KV cache + scheduler 假设是"单 model + 单 forward"，不适合

## 1. vLLM-Omni（vllm-project 官方）

### 项目状态

- 仓库：[vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni)
- **首发：2025.11**，**arxiv 论文**：[arXiv:2602.02204 · vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models](https://arxiv.org/abs/2602.02204)
- 4400+ star，180+ contributors（已是中型项目）
- 当前版本：v0.18.0（2026.3）— 强化核心 runtime + 统一量化 + diffusion 执行

### 核心设计

#### 1.1 Stage 抽象

把复杂的 omni 推理图建模为**有向图**：

```
nodes  = 模型组件（LLM、DiT、vision encoder、audio encoder...）
edges  = stage-transfer 函数（数据如何从一个 stage 流到下一个）
```

每个 stage 独立服务、独立调度、独立 batch。

#### 1.2 OmniConnector：跨 stage 的数据搬运

stage 之间通过 OmniConnector 做：

- KV cache 跨节点传输（基于 Mooncake Transfer Engine）
- AR/DiT 中间结果传递
- RDMA 优化（GPU Direct）

#### 1.3 EPDG 全分离

```
Encoder → Prefill → Decode → Generate (TTS/DiT)
   E         P         D          G
```

P0 优先级落在 Prefill-Decode 分离（Qwen2.5-Omni / 3-Omni），其他 stage 逐步推进。

### 为什么这是 2026 重点跟踪项目

- **承袭 vLLM 生态**（PagedAttention / continuous batching / OpenAI API 全部能用）
- **官方一体化**（不是个野生 fork）
- **同时覆盖 AR + DiT**（这是 SGLang 当前没做的）
- 性能数据：**JCT 比 baseline 降低高达 91.4%**

### 阅读路径（按顺序）

1. ⭐⭐⭐ [arXiv 论文](https://arxiv.org/abs/2602.02204) — 必读，理解 stage 抽象 + OmniConnector
2. ⭐⭐⭐ [官方 README](https://github.com/vllm-project/vllm-omni/blob/main/README.md) — 项目能力速览
3. ⭐⭐⭐ [Disaggregated Inference for Omni-Modality Models 设计文档](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/disaggregated_inference/) — 完整设计图
4. ⭐⭐⭐ [Q1 2026 Roadmap RFC #1192 · Omni Connector 全分离架构](https://github.com/vllm-project/vllm-omni/issues/1192) — 看官方下半年要做什么（这种 RFC 是研究选题灵感金矿）
5. ⭐⭐ [Qwen2.5-Omni 端到端示例代码](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/qwen2_5_omni/end2end.py) — 跑通需要这一段
6. ⭐⭐ [vLLM-Omni · Qwen2.5-Omni 部署文档](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/examples/offline_inference/qwen2_5_omni/) — 含 prompt 格式（`<\|audio_bos\|>`、`<\|AUDIO\|>`、`<\|vision_bos\|>` 等特殊 token）
7. ⭐⭐ [vLLM PR #26156 · Add support for audio in video in Qwen2.5-Omni](https://github.com/vllm-project/vllm/pull/26156) — 看官方怎么实现"video 里的 audio 流"
8. ⭐⭐ Jimmy Song《用 vLLM-Omni 快速部署多模态推理》— [jimmysong.io 中文](https://jimmysong.io/zh/ai/vllm-omni/) — 中文部署快速上手
9. ⭐ [Moonlight 论文评述](https://www.themoonlight.io/review/vllm-omni-fully-disaggregated-serving-for-any-to-any-multimodal-models) — 第三方学术综述

## 2. SGLang-Omni（sgl-project 官方）

### 项目状态

- 仓库：[sgl-project/sglang-omni](https://github.com/sgl-project/sglang-omni)
- 比 vLLM-Omni 更早开始 RFC 但发展更慢
- 77 star，20 contributors，活跃 RFC + 重构中
- 当前重点：Qwen3-Omni 集成 + 架构精简（重构为 Stage → Engine 两层）

### 核心设计

#### 2.1 Inter-Disaggregation（**跨 stage 分离**）

把整个 omni 模型切成多个独立 stage：

```
Encoder Stage  ←→  Thinker Stage  ←→  Talker Stage
       ↑                  ↑                  ↑
  独立进程            独立进程            独立进程
       └──── 消息队列通信 ─────┘
```

每个 stage 独立调度、独立 GPU 分配。

#### 2.2 Intra-Disaggregation（**stage 内分离**）

同一 stage 内部还可以再切：

- 不同 encoder 放在不同进程
- LLM 走 PD 分离（Prefill/Decode）

#### 2.3 多数据流路径支持

omni 模型的输入/输出组合是组合爆炸的：

```
text-only / text+vision / text+audio / text+vision+audio
   ↓           ↓             ↓             ↓
text out / text+audio out / text+image out / text+audio+image out
```

SGLang-Omni 用**统一 stage 框架**支持所有这些路径。

### 当前架构问题（**研究选题金矿**）

**Issue #188 Refactoring Proposal** 公开承认现有抽象**过度工程化**：

```
HTTP API → Client → Coordinator → Stage → Worker → Executor → Engine → Scheduler → ModelRunner
```

9 层！其中 Stage → Worker → Executor → Engine 四层职责重叠严重。**正在重构成 Stage → Engine 两层**。

### Qwen3-Omni 集成（已落地优化）

针对 Qwen3-Omni-Thinker 的关键优化（[Issue #92](https://github.com/sgl-project/sglang-omni/issues/92)）：

- **FusedMoE**：用单个 Triton kernel 替代 HuggingFace 分散式 MoE。Qwen3-Omni 是 64 expert / top-8 MoE，加速明显
- **VisionAttention**：Flash Attention + cu_seqlens，支持打包 batch、无 padding 开销
- **Tensor Parallelism**：列并行 + 行并行 linear 层

### Ming-Omni 集成（多 stage 范式样板）

[Issue #236](https://github.com/sgl-project/sglang-omni/issues/236) 展示了 6 阶段管道：

```
预处理 → 音频编码器 → 聚合 → Thinker → 解码/Talker（并行）
```

**Ming-Omni vs Qwen3-Omni 关键差异**：
- Qwen3-Omni：Thinker 和 Talker 通过**文本 token 串联**（松耦合）
- Ming-Omni：Thinker 和 Talker 通过**解码后的文本**串联（更紧耦合）

### 阅读路径

1. ⭐⭐⭐ [SGLang-Omni 主仓库 README](https://github.com/sgl-project/sglang-omni)
2. ⭐⭐⭐ [RFC #16546 · SGLang-Omni Design](https://github.com/sgl-project/sglang/issues/16546) — 整体设计哲学（inter/intra disaggregation）
3. ⭐⭐⭐ [Refactoring Proposal #188](https://github.com/sgl-project/sglang-omni/issues/188) — 看官方对现有架构的反思
4. ⭐⭐⭐ [Qwen3-Omni Refactor #92](https://github.com/sgl-project/sglang-omni/issues/92) — 具体模型集成
5. ⭐⭐ [Ming-Omni Support Design Spec #236](https://github.com/sgl-project/sglang-omni/issues/236) — 多 stage 管道实例
6. ⭐⭐ [Qwen3-Omni Technical Report (arXiv:2509.17765)](https://arxiv.org/abs/2509.17765) — Thinker-Talker 架构论文
7. ⭐⭐ [Multimodal Models · SGLang Docs](https://sgl-project.github.io/supported_models/text_generation/multimodal_language_models.html) — 当前支持模型列表

## 3. 两条路线对比（面试可讲）

| 维度 | vLLM-Omni | SGLang-Omni |
|---|---|---|
| 发布时间 | 2025.11 | 2024.末 RFC，2025 落地 |
| 当前活跃度 | 高（4400 star，月度 release） | 中（77 star，重构中） |
| 论文 | ✅ arXiv:2602.02204 | ❌（设计在 RFC 里） |
| 核心抽象 | Stage 图 + OmniConnector | Inter / Intra Disaggregation |
| 数据搬运 | Mooncake Transfer Engine + RDMA | 进程间消息队列 |
| AR / 非 AR | ✅ 同时支持（DiT 一等公民） | 主要 AR + TTS（DiT 弱） |
| 性能数据 | JCT ↓ 91.4% | 暂无公开 benchmark |
| 适合什么场景 | 大规模生产部署 | 研究 / 实验 / 自定义 omni 模型 |

**面试官追问预案**：

- Q：omni-modal 推理为什么不能直接用 vLLM？
  - A：① 单 model 假设；② KV cache 假设是 attention 的，DiT 没 KV；③ scheduler 假设单 forward；④ output modality 假设是 text token 流
- Q：vLLM-Omni 的 OmniConnector 解决什么问题？
  - A：跨 stage 的数据搬运，特别是 stage 在不同节点时（用 Mooncake Transfer Engine + RDMA）
- Q：SGLang-Omni 的 inter / intra disaggregation 区别？
  - A：inter 是把 omni 模型按 encoder/Thinker/Talker 切到不同进程；intra 是同一 stage 内部再切（如多个 encoder 各占一进程，或 LLM 内部 PD 分离）
- Q：为什么有了 vLLM-Omni 还需要 SGLang-Omni？
  - A：① 设计哲学不同（vLLM-Omni 偏生产、SGLang-Omni 偏研究）；② SGLang-Omni 更易做自定义 stage 组合；③ 和 SGLang 主框架的 RadixAttention 有协同收益

## 4. 给你的实际行动建议

### 方案 A：W23 Day 5 浅尝（推荐）

只在 W23 最后一天读上面"阅读路径"的前 3 篇（每个项目）。重点记住：

- 这俩项目存在
- 它们解决什么问题（vLLM/SGLang 主项目解决不了）
- 关键术语：Stage 抽象、OmniConnector、Inter/Intra Disaggregation、Thinker-Talker

简历可以写一行："了解 vLLM-Omni / SGLang-Omni 等 omni-modal 推理系统的 disaggregated 设计哲学"。

### 方案 B：W23 + W24 buffer 时间深挖（如果时间够）

加做：

- 跑通 vLLM-Omni 上的 Qwen2.5-Omni（需要 H100/H200，5090 不够）
- 给 SGLang-Omni 提一个文档 PR（`good first issue` 标签里有）
- 写一篇博客《2026 omni-modal 推理两大流派对比》

### 方案 C：作为研究选题（**仅限论文方向需要扩展时**）

vLLM-Omni 的 Q1 2026 Roadmap RFC #1192 列出了一堆"待做"项，每一个都是潜在论文方向。但这超出 6 个月学习计划的范围，看你硕士论文那个窗口里要不要捡。

## 5. 我的核心理解（用自己话写）

```
（vLLM-Omni 是把"多模态推理"建模成 dataflow 图，每个 node 是 model，每个 edge 是 connector
 SGLang-Omni 是把"omni 模型"切到多个独立 stage 进程，用消息队列通信
 两者都在解决同一个问题：单 forward / 单 KV cache 假设在 omni 模型上不成立
 哪个会赢？取决于谁先把 DiT 类非 AR 模型做透，谁就吃下视频生成这个未来主战场）
```
