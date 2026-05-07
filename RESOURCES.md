# 总资源清单

按主题分类的所有学习资料链接。每个阶段的 `resources.md` 是这一份的子集。

---

## 收录与排序原则（v2 优中选优）

> 之所以做这次整改：第一版资料是"广撒网"，不少二手转载和缺乏作者背景的文章混在里面。研究类学习不能这样。

**采纳标准**：

1. **作者可考据** —— 公司/职位/历史一致性，非匿名号
2. **时效性** —— 内容近 12 个月仍准确（被新模型推翻的旧文降级或剔除）
3. **评论区质量** —— 无重大未回应技术质疑
4. **优先含**公式推导 / 源码引用 / 复现实验，避免纯翻译型
5. **每条都标注** `[作者背景, 平台 YYYY.MM]`，让你自评权威度和时效

**星级**：

- ⭐⭐⭐ 必读（强烈推荐）
- ⭐⭐ 推荐（认真读完）
- ⭐ 选读（按需）

**新增条目时请遵守同样规则**。如果不确定一篇文章是否合格，宁缺毋滥。

---

## 1. Transformer / Attention 基础（M1 用）

### 注意力变体（MHA → MQA → GQA → MLA → DSA → CSA+HCA）

#### 苏剑林系列（**必读，按时间顺序**）

- ⭐⭐⭐ 苏剑林《缓存与效果的极限拉扯：从 MHA、MQA、GQA 到 MLA》— [追一科技算法研究员 / kexue.fm 2024.5](https://kexue.fm/archives/10091) — 中文圈最权威的 attention 演进综述
- ⭐⭐⭐ 苏剑林《Transformer 升级之路：20、MLA 好在哪里？(上)》— [kexue.fm 2025.5](https://kexue.fm/archives/10907) — **必加**！这是上一篇的续作，作者本人和 DeepSeek 团队讨论过 MLA。新论点："MLA 本质是 NoPE 的 MHA，训练时 head_dims=192，decode 时变 MQA-512"

#### zartbot 系列（**国内一线推理工程师，DeepSeek 团队认证过**）

- ⭐⭐ zartbot《从 MHA 到 MLA 看 Attention 优化》— [微信公众号 2024.5](https://mp.weixin.qq.com/s/l2uUXGQ-8Rj_nI3JG5lZ_g) — 从代数视角看 MLA 的低秩分解
- ⭐⭐ zartbot《继续谈谈 MLA 以及 DeepSeek-MoE 和 SnowFlake Dense-MoE》— [微信公众号 2024.5](https://mp.weixin.qq.com/s/hI7q4_-ZMtFIQ-ckhM9_YQ) — 配 SnowFlake Dense-MoE 对比，参数量分析（MLA 的 KV 参数量仅为 MHA 的 11%）
- ⭐⭐ zartbot《详细谈谈 DeepSeek MoE 相关的技术发展》— [微信公众号 2025.2](https://mp.weixin.qq.com/s/WFJxnTF9fGIIXPA7GQ5V2w) — 文中明确说"得到了 DeepSeek 团队同学的指正"，权威性强

#### 综述

- ⭐⭐ Yue Shui《Transformer 注意力机制：MHA、MQA 与 GQA 的对比》— [syhya.github.io 2025.1](https://syhya.github.io/zh/posts/2025-01-16-group-query-attention/) — 配公式 + 图，适合跟着推导

### 源码仓库（顺序读）

1. `karpathy/nanoGPT` — 500 行 GPT-2 复现，标杆教学代码
2. `meta-llama/llama` — `model.py` 中的 GQA + RoPE 实现
3. `deepseek-ai/DeepSeek-V3` — `model.py` 中的 MLA 实现
4. `deepseek-ai/DeepSeek-V3.2-Exp` — DSA Lightning Indexer
5. `QwenLM/Qwen3` — Gated DeltaNet 混合实现

### 视频（B 站）

- 李宏毅《生成式 AI 导论》Transformer 章节（中文，自带字幕）
- 李沐《动手学深度学习》Transformer 章节
- 王树森《Transformer 模型》系列

---

## 2. GPU 编程：CUDA + Triton（M2 用）

### CUDA 入门

- ⭐⭐⭐ **PMPP 第 4 版**（Programming Massively Parallel Processors）— 教材
- ⭐⭐⭐ 李理《PMPP 第四版》中文翻译 — [fancyerii.github.io 2024.2](https://fancyerii.github.io/2024/02/20/pmpp/) — **网上最完整的 PMPP 中文版**，覆盖 22 章
- ⭐⭐⭐ smarterhhsu/PMPP-Learning — [GitHub](https://github.com/smarterhhsu/PMPP-Learning) — 38 个 Exercise CUDA 实现 + 中文笔记
- ⭐⭐ Smarter《PMPP 导读》— [smarter.xin 2024](https://smarter.xin/posts/30730973/) — 学习路线 + 各章概览

### CUDA 实战 / 高阶

- ⭐⭐⭐ BBuf [`how-to-optim-algorithm-in-cuda`](https://github.com/BBuf/how-to-optim-algorithm-in-cuda) — [OneFlow 工程师维护的 GitHub 仓库] — 国内 CUDA/MLSys 学习圣地
- ⭐⭐⭐ BBuf [`RESOURCES.md`](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/master/RESOURCES.md) — 这一份本身就是国内 CUDA/Triton/MLSys 资源最全索引，**直接挂这个链接，让你能继续向下挖**
- ⭐⭐ CUDA-MODE 课程笔记系列（B 站搜 "CUDA-MODE 课程笔记"）— BBuf 等人翻译的 GPU MODE 课程

### Triton

- ⭐⭐⭐ BBuf《OpenAI Triton 入门笔记一/二/三》— [微信公众号系列 2024](https://github.com/BBuf/how-to-optim-algorithm-in-cuda) — Triton 入门最佳中文资料
- ⭐⭐ Triton 官方教程 — [triton-lang.org](https://triton-lang.org/main/getting-started/tutorials/) — 官方 6 个 tutorial（vector add → softmax → matmul → fused attention）

### FlashAttention（**M2 W8 主菜**）

- ⭐⭐⭐ 猛猿《图解大模型计算加速系列：FlashAttention V1，从硬件到计算逻辑》— [知乎 2023.11](https://zhuanlan.zhihu.com/p/669926191) — **BBuf 在自己 Triton 笔记里反复推荐**，中文圈 FA1 解读最权威
- ⭐⭐⭐ 猛猿《图解大模型计算加速系列：Flash Attention V2，从原理到并行计算》— [微信公众号 2023.7](https://mp.weixin.qq.com/s/5K6yNj23NmNLcAQofHcT4Q) — 同作者，FA2 配套
- ⭐⭐ 《FlashAttention 算法之美：极简推导版》— [微信公众号](https://mp.weixin.qq.com/s/hu5D1dmCFkeStxbXBE-czA) — 简化版，适合入门第一遍
- ⭐⭐ Zihao Ye《From Online Softmax to FlashAttention》（英文）— FlashInfer 作者写的完整推导，进阶必读

---

## 3. vLLM / SGLang 推理框架（M3-M4 用）

### vLLM 源码地图（**M3 主菜**）

- ⭐⭐⭐ Smarter《vLLM 系统拆解》系列 — [smarter.xin 2025.10-11 持续更新](https://smarter.xin/posts/354d88e4/) — **七层结构图是教科书级**，按 vLLM 官方架构文档对齐，面试问到某个能力落在哪层时直接能答
- ⭐⭐⭐ shijuzhao《vLLM V1: 大模型推理系统的教科书》— [shijuzhao.github.io 2025.2](https://shijuzhao.github.io/vllm-v1) — **基于 v1 0.7.3 版本逐行代码解读，最完整的 V1 中文资料**
- ⭐⭐⭐ 猛猿《图解大模型计算加速系列：vLLM 源码解析 1-N，整体架构》— [微信公众号系列 2024](https://mp.weixin.qq.com/s/r_t6_zMvPT7za82MZX4oRA) — 多篇连载，强调"图解 + 整体架构"
- ⭐⭐ "Inside vLLM: Anatomy of a High-Throughput LLM Inference System" 中文版 — [知乎 2025](https://zhuanlan.zhihu.com/p/2021773453523522489) — 翻译自 vLLM 官方博客

### vLLM 原理 + 调优

- ⭐⭐ quant67《大模型基础设施工程》系列 — [quant67.com 2025](https://quant67.com/post/llm-infra/12-paged-continuous/12-paged-continuous.html) — 从 PagedAttention 到 vLLM v1 全覆盖，含调参手册

### B 站源码课

- 搜 "vLLM 源码全流程分析"（lyrry1997 系列，带飞书课件）
- 搜 "vLLM 大模型推理框架 分块显存管理"

### SGLang vs vLLM 横向对比（M4 用）

- ⭐⭐⭐ SGLang 官方 RadixAttention 文档 — [docs.sglang.io](https://mintlify.com/sgl-project/sglang/concepts/radix-attention) — 含 `radix_cache.py` 关键源码片段
- ⭐⭐⭐ LMSYS 官方博客《Fast and Expressive LLM Inference with RadixAttention and SGLang》— [lmsys.org 2024.1](https://lmsys.org/blog/2024-01-17-sglang/) — 原作者团队解读
- ⭐⭐《RadixAttention 技术详解：从原理到 SGLang 实践及 vLLM APC 对比》— [华为昇腾开源专区 2026.3](https://ascendai.csdn.net/69c390dd54b52172bc642216.html) — 含 SGLang 源码 + 与 vLLM Automatic Prefix Caching 对比

### 工程优化武器库（M4 用）

- 投机采样：搜 "投机采样综述 中文"、"Medusa EAGLE MTP 对比"
- 量化：搜 "AWQ 论文解读 中文"、"FP8 训练 DeepSeek"

---

## 4. 推理系统架构 · PD/AF/EP 分离 + KV cache 存储 + Kernel 库（**横跨 M3-M5 的纵向主题**）

> 完整深度专题见 [`M4-inference-frameworks/w16-disaggregated-prefill/notes-deep.md`](./M4-inference-frameworks/w16-disaggregated-prefill/notes-deep.md)
> 这一节涵盖 PD 分离、AF 分离、大 EP/EPLB、xPyD 弹性 PD、FlashInfer 等 2025-2026 推理 infra 最热的系统级方向。

### 4.1 PD 分离 · Mooncake（**论文一作章明星本人写的解读**）

- ⭐⭐⭐ 章明星《Mooncake (1)：在月之暗面做月饼，Kimi 以 KVCache 为中心的分离式推理架构》— [清华助理教授 + 月之暗面 KVCache.AI 团队负责人 / 2024.6](https://www.163.com/dy/article/J64853CG055689ZC.html) — 含原论文未提的 design choice 思考
- ⭐⭐ 《对话清华章明星、月之暗面许欣然：Mooncake 架构背后》— [硅星人 ACC 2024.11](https://view.inews.qq.com/a/20241121A02FF200) — 用"备菜/炒菜"类比讲清 PD 分离
- ⭐⭐《Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving 阅读笔记》— [掘金 2025](https://juejin.cn/post/7510051346306105394) — 调度算法详解
- ⭐⭐⭐ Mooncake 论文 [arXiv:2407.00079](https://arxiv.org/abs/2407.00079)

### 4.2 xPyD 弹性 PD（**2026 主流落地**）

- ⭐⭐⭐ [vLLM PR #18242 · xPyD based on P2P NCCL](https://github.com/vllm-project/vllm/pull/18242) — vLLM xPyD 核心实现
- ⭐⭐⭐ [vLLM PR #12957 · XpYd with MooncakeStore](https://github.com/vllm-project/vllm/pull/12957) — Mooncake backend
- ⭐⭐⭐ [SGLang PD Disaggregation 官方文档（中文）](https://docs.sglang.com.cn/advanced_features/pd_disaggregation.html) — 完整启动命令
- ⭐⭐ [SGLang Issue #9442 · xPyD 配置示例](https://github.com/sgl-project/sglang/issues/9442)

### 4.3 AF 分离（Attention/FFN Disaggregation）—— **2026 新方向**

- ⭐⭐⭐ **MegaScale-Infer (字节跳动, arXiv:2504.02263)** — 为 MoE 提供 AF 分离 + Ping-pong 流水线 + M2N 通信，**+1.90× 吞吐**
- ⭐⭐⭐ [Theoretically Optimal A/F Ratios (arXiv:2601.21351)](https://arxiv.org/html/2601.21351v1) — 理论最优 A/F 比例分析
- ⭐⭐⭐ [百度 AFD 死区分析 (arXiv:2602.09721)](https://arxiv.org/pdf/2602.09721) — "死区"现象 + AFD vs EP 对比
- ⭐⭐⭐ [vLLM RFC #22799 · ATTN-FFN Disaggregation for MoE Models](https://github.com/vllm-project/vllm/issues/22799) — **2025.8 启动，2026 重点跟踪**

### 4.4 大 EP / EPLB（**MoE 推理标配**）

- ⭐⭐⭐ [DeepSeek EPLB GitHub](https://github.com/deepseek-ai/EPLB) — DeepSeek 开源 Expert Parallelism Load Balancer
- ⭐⭐⭐ [腾讯云 · EP 架构：DeepSeek 突破性实践背后](https://cloud.tencent.com.cn/developer/article/2504080) — EP 终极形态之争
- ⭐⭐⭐ [腾讯云 · 如何重现 DeepSeek 推理性能突破](https://cloud.tencent.com/developer/article/2523038) — RTP-LLM 阿里云灵骏实测：Prefill 42.6K TPS / Decode 14.7K TPS
- ⭐⭐ [阿里云 · DeepSeek EPLB 冗余专家策略](https://developer.aliyun.com/article/1654261)

### 4.5 LMCache + 多级 KV cache

- ⭐⭐⭐ [LMCache 官方文档 · Architecture](https://docs.lmcache.ai/developer_guide/architecture.html) — GPU/CPU/Disk/Remote 四级架构图
- ⭐⭐ [LMCache CPU RAM 配置](https://docs.lmcache.ai/kv_cache/cpu_ram.html)
- ⭐⭐ [LMCache + Mooncake 集成](https://docs.lmcache.ai/kv_cache/storage_backends/mooncake.html)

### 4.6 FlashInfer · 推理 attention kernel 标杆库

- ⭐⭐⭐ [flashinfer-ai/flashinfer GitHub](https://github.com/flashinfer-ai/flashinfer) — Zihao Ye（FA 论文作者）维护
- ⭐⭐ [FlashInfer 源码级解读](https://www.yeyulingfeng.com/534685.html) — 中文源码导读
- 关键能力：稀疏注意力 90% 带宽 + JIT 自定义 + PageAttention + Top-K 采样 kernel
- 用途：vLLM v1 默认 backend、SGLang `fa3` backend、DeepSeek V4 SGLang 部署 backbone

---

## 5. 分布式并行 + 训练框架（M5 上半 用）

### 5.1 Megatron 5 大并行

- ⭐⭐ [Megatron-LM 深度解析 5 大并行](https://adg.csdn.net/6952548a5b9f5f31781b8f89.html) — 火山引擎 ADG 社区
- ⭐⭐ [MLTalks 详解 Megatron Pipeline Parallel](https://www.mltalks.com/posts/3278488319/) — GPipe / PipeDream / 1F1B / Interleaved 演进
- ⭐⭐ [幻方 · 模型并行 · Megatron 调优实验](https://www.high-flyer.cn/en/blog/model_parallel-1/inidex/) — 幻方 AI 实测数据
- 视频：B 站搜 "Megatron-LM 5 大并行策略"

### 5.2 三大训练框架横向对比（**FSDP vs DeepSpeed vs Megatron**）

> 完整选型决策树见 [`M5/w17-w18-parallelism/notes/08-fsdp-deepspeed-megatron.md`](./M5-parallelism-architectures/w17-w18-parallelism/notes/08-fsdp-deepspeed-megatron.md)

- ⭐⭐⭐ **[quant67 · Megatron-LM 与 DeepSpeed](https://quant67.com/post/llm-infra/07-megatron-deepspeed/07-megatron-deepspeed.html)** — **最完整的中文对比**，含三流派演进历史 + 规模选型表
- ⭐⭐⭐ **[HuggingFace · 从 DeepSpeed 到 FSDP 再回到 Accelerate](https://huggingface.co/blog/zh/deepspeed-to-fsdp-and-back)** — HF 官方实战，含 DeepSpeed 强制 FP32 vs FSDP 默认精度的关键差异
- ⭐⭐ [HuggingFace · FSDP vs DeepSpeed 概念指南](https://hugging-face.cn/docs/accelerate/concept_guides/fsdp_and_deepspeed)
- ⭐⭐ [大模型并行训练指南：通俗理解 Megatron-DeepSpeed](https://blog.csdn.net/v_JULY_v/article/details/132462452) — v_JULY_v 长文

#### 论文

- ⭐⭐⭐ ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (SC'20)
- ⭐⭐⭐ Megatron-LM (arXiv:1909.08053) + Megatron-LM 2 (arXiv:2104.04473) — 3D 并行
- ⭐⭐⭐ PyTorch FSDP (VLDB'24)

### 5.3 Muon 优化器（DeepSeek V4 已采用）

- 搜 "Muon 优化器解读"、"Newton-Schulz 迭代 大模型训练"、"Kimi Muon 论文"

---

## 6. 2026 最新架构（M5 下半 用 · 必修）

### DeepSeek 系（压缩稀疏全注意力路线）

#### V3 / V3.2-Exp（必修前置）

- ⭐⭐⭐ DeepSeek-V3 / V3.2-Exp 官方技术报告 PDF（HuggingFace 仓库）
- ⭐⭐ [52nlp · DeepSeek-V3.2-Exp 用稀疏注意力实现高效长上下文](https://www.52nlp.cn/deepseek-v3-2-exp%ef%bc%9a%e7%94%a8%e7%a8%80%e7%96%8f%e6%b3%a8%e6%84%8f%e5%8a%9b%e5%ae%9e%e7%8e%b0%e6%9b%b4%e9%ab%98%e6%95%88%e7%9a%84%e9%95%bf%e4%b8%8a%e4%b8%8b%e6%96%87%e6%8e%a8%e7%90%86)
- ⭐⭐ [机器之心 · DSA 公开解读](https://mp.weixin.qq.com/s/WYze9rEZnuZ9l1Y132VJmA)
- ⭐⭐ [CSDN · DSA 算法源码分析](https://deepseek.csdn.net/69f0799e54b52172bc7087e5.html)
- 📦 [deepseek-ai/DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)

#### V4（2026.4.24，必修）

- ⭐⭐⭐ DeepSeek-V4 官方技术报告 PDF（HuggingFace `deepseek-ai/DeepSeek-V4` 仓库）
- ⭐⭐⭐ [SGLang 官方 cookbook · DeepSeek-V4](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4) — Day-0 部署文档，含完整启动命令和硬件要求
- ⭐⭐⭐ [LMSYS 博客 · DeepSeek-V4 on Day 0: From Fast Inference to Verified RL with SGLang and Miles](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/) — SGLang 团队对 V4 hybrid attention + mHC + Muon 的深度技术解读
- ⭐⭐ [量子位 · V4 报告太详尽了 · 484 天换代之路](https://www.163.com/dy/article/KRBULUJ60511DSSR.html)
- ⭐⭐ [CSDN · DeepSeek-V4 技术报告全解读](https://deepseek.csdn.net/69f040bc0a2f6a37c5a685c2.html)
- ⭐⭐ [掘金 · V4 简要解读 · 含详细参数对比表](https://juejin.cn/post/7631898635937497134)

### Qwen / Kimi 系（门控线性混合路线）

#### Qwen3 / Qwen3-Next（前置）

- ⭐⭐ [Qwen3 技术报告详解](https://blog.csdn.net/kebijuelun/article/details/148070438)
- ⭐⭐ [Qwen3 技术报告解读](https://mp.weixin.qq.com/s/4Qz5CB4Upns6rW2jDSJ6gA)

#### Qwen3.6-35B-A3B / Qwen3.6-27B（2026.4.16，必修）

- ⭐⭐⭐ [SGLang 官方 cookbook · Qwen3.6](https://cookbook.sglang.io/autoregressive/Qwen/Qwen3.6) — Day-0 部署文档，含 Mamba Radix Cache 等 hybrid 模型特有调度策略
- ⭐⭐⭐ [HuggingFace · Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) — 模型卡 + 官方部署命令（vLLM ≥ 0.19.0 / SGLang ≥ 0.5.10）
- ⭐⭐ [博客园 · Qwen3.6-35B-A3B 全面评测](https://www.cnblogs.com/sing1ee/p/19885253)
- ⭐⭐ [Qwen3.5/3.6 混合注意力解析 · Gated DeltaNet + MoE 部署](https://dev.tekin.cn/blog/qwen3-5-hybrid-attention-gated-deltanet-moe-deployment)

#### Kimi K2 Thinking / Kimi Linear / MiniMax M2

- ⭐⭐⭐ [LLM 架构新趋势 · Kimi K2 Thinking 和 MiniMax-M2 之后是什么](https://devpress.csdn.net/v1/article/detail/154573876) — **必读综述**，把"线性 attention vs 全注意力"博弈讲清楚
- ⭐⭐ [开源推理模型巅峰对决 · Kimi K2 Thinking vs MiniMax M2](https://blog.gitcode.com/f044616c21b80a6f28db2249be988908.html)
- ⭐⭐ [Kimi K2 Thinking vs MiniMax M2 全面对比](https://kimi-k2.org/zh/blog/17-kimi-k2-thinking-vs-minimax-m2)

---

## 7. RL 训推一体（M6 上半 用）

### 视觉化教学（**先看这个建立直觉**）

- ⭐⭐⭐ changyeyu/LLM-RL-Visualized — [GitHub](https://github.com/changyeyu/LLM-RL-Visualized) — 100+ 原创架构图覆盖 LLM/VLM/RL/PPO/GRPO/DPO，作者出版了豆瓣高分书《大模型算法：强化学习、微调与对齐》，权威性极高

### R1 复现教程（最佳入门实战）

- ⭐⭐⭐ [Datawhale R1 中文复现教程](https://cloud.tencent.com/developer/article/2494040) — 3×A800 跑通 GRPO，约 420 元
- ⭐⭐ [HuggingFace · Mini-R1 复现 DeepSeek R1 灵光一现](https://hugging-face.cn/blog/open-r1/mini-r1-contdown-game)

### 算法解读

- ⭐⭐⭐ 《大模型后训练中的 GRPO 算法剖析》— [知乎 2025](https://zhuanlan.zhihu.com/p/2020218462119679502) — 从"为什么 LLM 后训练适合 GRPO"角度讲清楚 PPO → GRPO 演进
- ⭐⭐ [柠檬 CC · RLHF 实战系列](https://www.limoncc.com/post/47f9ae3708f426c0/) — PPO/GRPO/REINFORCE++ 公式推导

### 框架

- ⭐⭐⭐ [OpenRLHF README 中文版](https://github.com/OpenRLHF/OpenRLHF/blob/main/README_zh.md) — Ray + vLLM + DeepSpeed 工业级 RL 框架
- 📦 字节 veRL · `volcengine/verl`
- 📦 HuggingFace TRL · `huggingface/trl`

---

## 8. ViT / 多模态 / DiT（M6 中段 用）

### Qwen2.5-VL

- ⭐⭐ [博客园 · Qwen2.5-VL 技术报告精读](https://www.cnblogs.com/emergence/p/18873748)
- ⭐⭐ [Qwen2.5-VL 算法解析](https://jishuzhan.net/article/2045498484711817217)
- ⭐⭐ [《Qwen2.5-VL》论文精读笔记](https://www.wsisp.com/helps/55456.html)

### InternVL

- ⭐⭐ [InternVL-2.5 训练细节](https://devpress.csdn.net/v1/article/detail/144970962) — 动态分辨率切片

### DiT

- 论文：Scalable Diffusion Models with Transformers
- 知乎搜 "DiT 详解"

### Omni-Modal 推理系统（**2026 上半年新热点 · 简历加分项**）

> 完整阅读路线 + 面试 Q&A 见 [`M6-rl-multimodal-interview/w23-multimodal/notes/05-omni-modal-systems.md`](./M6-rl-multimodal-interview/w23-multimodal/notes/05-omni-modal-systems.md)
> 收录这两个项目的理由：vLLM 和 SGLang 各自发布的**官方独立子项目**，专门解决"omni-modal（任意模态进任意模态出）"推理的新挑战 —— 现有 vLLM/SGLang 主项目的"单 model + 单 forward"假设不再成立。

**vLLM-Omni（vllm-project 官方，4400+ star）**

- ⭐⭐⭐ **[arXiv:2602.02204 · vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models](https://arxiv.org/abs/2602.02204)** — [vLLM 团队 / arXiv 2026.2] — 论文必读，Stage 抽象 + OmniConnector + EPDG 全分离
- ⭐⭐⭐ [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni) — [vllm-project 官方 / 2025.11 首发, v0.18.0 (2026.3)] — 4400+ star，180+ contributors
- ⭐⭐⭐ [Q1 2026 Roadmap RFC #1192 · Omni Connector 全分离架构](https://github.com/vllm-project/vllm-omni/issues/1192) — 是研究选题金矿
- ⭐⭐ [Disaggregated Inference for Omni-Modality Models 设计文档](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/disaggregated_inference/)
- ⭐⭐ Jimmy Song《用 vLLM-Omni 快速部署多模态推理》— [jimmysong.io 中文](https://jimmysong.io/zh/ai/vllm-omni/) — 中文部署快速上手

**SGLang-Omni（sgl-project 官方）**

- ⭐⭐⭐ [sgl-project/sglang-omni](https://github.com/sgl-project/sglang-omni) — [sgl-project 官方] — Inter/Intra-Disaggregation
- ⭐⭐⭐ [RFC #16546 · SGLang-Omni Design](https://github.com/sgl-project/sglang/issues/16546) — 整体设计哲学
- ⭐⭐⭐ [Refactoring Proposal #188](https://github.com/sgl-project/sglang-omni/issues/188) — 官方对现有架构的反思（**真实 infra 演进案例**）
- ⭐⭐ [Qwen3-Omni Refactor #92](https://github.com/sgl-project/sglang-omni/issues/92) — FusedMoE / VisionAttention 优化
- ⭐⭐ [Ming-Omni Support Design Spec #236](https://github.com/sgl-project/sglang-omni/issues/236) — 多 stage 管道实例

**Omni 模型本身（论文）**

- ⭐⭐⭐ [Qwen3-Omni Technical Report (arXiv:2509.17765)](https://arxiv.org/abs/2509.17765) — Thinker-Talker MoE 架构必读

---

## 9. 面试八股（M6 末段 用）

- ⭐⭐⭐ [小林 coding · 530+ 大模型面试题](https://xiaolincoding.com/other/ai.html)
- ⭐⭐ [字节大模型一面真题](https://blog.csdn.net/2401_84033492/article/details/141093004)
- ⭐⭐ [字节大模型一二三面经](https://www.nowcoder.com/feed/main/detail/ef14567a048e4e3fa30cbe28eb5c59e3)
- ⭐⭐ [入职字节大模型岗面试分享](https://devpress.csdn.net/v1/article/detail/147386717)
- LeetCode hot 100

---

## 附录 A · 两大 SOTA 资源库（持续向下挖）

这两份是国内 ML Sys / 推理 infra 圈最有价值的元资源库，挂这里让你能持续发现新资料：

- ⭐⭐⭐ **BBuf [`how-to-optim-algorithm-in-cuda/RESOURCES.md`](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/master/RESOURCES.md)** — OneFlow 工程师维护，CUDA/Triton/MLSys 资源最全索引，每月更新
- ⭐⭐⭐ **[`zhaochenyang20/Awesome-ML-SYS-Tutorial`](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial)** — SGLang 团队成员维护的 ML Sys 资源库，含 SGLang 源码深度解读

## 附录 B · 我额外推荐你订阅的渠道

### 公众号 / 博客（每天 / 每周打开）

- **机器之心**（公众号）：每天看新模型发布
- **量子位**（公众号）：业界动态
- **BBuf 的 CUDA 笔记**（公众号）：CUDA/Triton 实战
- **zartbot**（公众号）：DeepSeek/MoE/MLA 一线实战
- **Smarter's blog**（[smarter.xin](https://smarter.xin/)）：vLLM/PMPP 系列
- **科学空间**（[kexue.fm](https://kexue.fm)）：苏剑林，注意力数学原理

### 推特 / X（开 VPN 看）

- @lvwerra（HuggingFace TRL 作者）
- @woosuk_k（vLLM 主创）
- @teknium1
- @karpathy

### GitHub Trending

- 每周看一次 `Trending Python / C++ / CUDA`，找新出的推理优化项目

---

## 附录 C · 已剔除清单（避免共创同学重复加）

下面这些大热文章被我们剔除了，原因如下，如果你有反对意见可以 PR 讨论：

| 文章 / 来源 | 剔除原因 |
|---|---|
| 借一步网《缓存与效果的极限拉扯》 | 纯转载苏剑林 kexue.fm 原文，无附加价值 |
| 冷眸《Attention 各种变体》 | 内容混合多家观点但无独立洞察 |
| U 深研《国内大模型厂商混合注意力机制》 | 偏新闻聚合性质，技术深度不够 |
| 53AI《大模型推理框架 vLLM 源码解析》 | 与猛猿原文重合度极高，二手转载 |
| 一些匿名公众号"翻译版" / "速读版" FA 文章 | 来源不可考，公式有误的概率高，建议直接读猛猿原版 |

## 附录 D · 加新条目时请遵守的格式

```
- ⭐⭐⭐ 标题 — [作者背景, 平台 YYYY.MM](URL) — 一句话价值说明（必须）
                ↑                              ↑
        让读者判断权威度                让读者知道为什么收
```

新人贡献流程：在你的个人分支加资源，发 PR 到 main 时附理由（作者是谁、为什么列入、星级依据）。如果 reviewer 觉得不达标，会请你降级或剔除，不要难过 —— 是为了保护清单的信噪比。
