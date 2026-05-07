# M5 资源清单（顶层导航）

> 收录原则见顶层 [RESOURCES.md](../RESOURCES.md) 的"收录与排序原则"。

## W17-W18 · 并行（详见子目录 `w17-w18-parallelism/`）

### Megatron 5 大并行

- ⭐⭐ [Megatron-LM 深度解析 5 大并行](https://adg.csdn.net/6952548a5b9f5f31781b8f89.html) — 火山引擎 ADG 社区
- ⭐⭐ [MLTalks · 详解 Megatron Pipeline Parallel](https://www.mltalks.com/posts/3278488319/) — GPipe / PipeDream / 1F1B / Interleaved 演进
- ⭐⭐ [幻方 · 模型并行 · Megatron 调优实验](https://www.high-flyer.cn/en/blog/model_parallel-1/inidex/) — 幻方 AI 实测数据
- ⭐⭐ [Transformer 巨型模型训练核心技术 · Megatron-LM](https://blog.csdn.net/qq_22409661/article/details/145790287)
- 视频：B 站搜 "Megatron-LM 5 大并行策略"

### 训练框架横向对比（**FSDP vs DeepSpeed vs Megatron**）

> 完整对比 + 选型建议见笔记 [`w17-w18-parallelism/notes/08-fsdp-deepspeed-megatron.md`](./w17-w18-parallelism/notes/08-fsdp-deepspeed-megatron.md)

- ⭐⭐⭐ **[quant67 · Megatron-LM 与 DeepSpeed](https://quant67.com/post/llm-infra/07-megatron-deepspeed/07-megatron-deepspeed.html)** — [土法炼钢兴趣小组] — **最完整的中文对比**，含三流派演进历史和规模选型建议
- ⭐⭐⭐ **[HuggingFace · 从 DeepSpeed 到 FSDP 再回到 Accelerate](https://huggingface.co/blog/zh/deepspeed-to-fsdp-and-back)** — HF 官方实战，含 DeepSpeed 强制 FP32 优化器 vs FSDP 默认精度的差异分析
- ⭐⭐ [HuggingFace · FSDP vs DeepSpeed 概念指南](https://hugging-face.cn/docs/accelerate/concept_guides/fsdp_and_deepspeed)
- ⭐⭐ [大模型并行训练指南：通俗理解 Megatron-DeepSpeed](https://blog.csdn.net/v_JULY_v/article/details/132462452) — v_JULY_v 长文
- ⭐⭐ [Distributed Training: DeepSpeed ZeRO 1/2/3 + Accelerate, Megatron-LM](https://www.cnblogs.com/forhheart/p/18401234)

#### 论文

- ⭐⭐⭐ ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (SC'20)
- ⭐⭐⭐ Megatron-LM (arXiv:1909.08053)
- ⭐⭐⭐ Megatron-LM 2 (arXiv:2104.04473) — 3D 并行
- ⭐⭐⭐ PyTorch FSDP (VLDB'24)

### Muon 优化器（V4 已采用，2026 新八股）

- 搜 "Muon 优化器解读"
- 搜 "Newton-Schulz 迭代 大模型训练"
- Kimi Moonlight 论文（Muon 在大规模 MoE 上首次应用）

## W19-W20 · 新架构（详见子目录 `w19-w20-architectures/`）

### DeepSeek 系（V3 → V3.2 → V4，**压缩稀疏全注意力路线**）

#### V3 / V3.2-Exp（必修前置）

- ⭐⭐⭐ DeepSeek-V3 / V3.2-Exp 官方技术报告 PDF（HuggingFace 仓库）
- ⭐⭐ [52nlp · DeepSeek-V3.2-Exp 用稀疏注意力实现高效长上下文](https://www.52nlp.cn/deepseek-v3-2-exp%ef%bc%9a%e7%94%a8%e7%a8%80%e7%96%8f%e6%b3%a8%e6%84%8f%e5%8a%9b%e5%ae%9e%e7%8e%b0%e6%9b%b4%e9%ab%98%e6%95%88%e7%9a%84%e9%95%bf%e4%b8%8a%e4%b8%8b%e6%96%87%e6%8e%a8%e7%90%86)
- ⭐⭐ [机器之心 · DSA 公开解读](https://mp.weixin.qq.com/s/WYze9rEZnuZ9l1Y132VJmA)
- ⭐⭐ [CSDN · DSA 算法源码分析](https://deepseek.csdn.net/69f0799e54b52172bc7087e5.html)
- 📦 [deepseek-ai/DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)

#### V4（2026.4.24，必修）

- ⭐⭐⭐ DeepSeek-V4 官方技术报告 PDF（HuggingFace `deepseek-ai/DeepSeek-V4` 仓库）
- ⭐⭐⭐ **[SGLang 官方 cookbook · DeepSeek-V4](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4)** — Day-0 部署文档，含完整启动命令、各种硬件配置（H200 / B200 / GB200 / GB300）和三种 serving recipe（low-latency / balanced / max-throughput）
- ⭐⭐⭐ **[LMSYS 博客 · DeepSeek-V4 on Day 0: From Fast Inference to Verified RL with SGLang and Miles](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)** — SGLang 团队对 V4 hybrid attention + mHC + Muon + FP4 的深度技术解读，含 ShadowRadix prefix cache、HiSparse CPU-extended KV、TileLang mHC 等核心 kernel 细节
- ⭐⭐ [量子位 · V4 报告太详尽了 484 天换代](https://www.163.com/dy/article/KRBULUJ60511DSSR.html)
- ⭐⭐ [CSDN · V4 技术报告全解读](https://deepseek.csdn.net/69f040bc0a2f6a37c5a685c2.html)
- ⭐⭐ [掘金 · V4 简要解读 + 详细参数表](https://juejin.cn/post/7631898635937497134)
- ⭐⭐ vLLM V4 PR：[#40860 (主体)](https://github.com/vllm-project/vllm/pull/40860) + [#40899 (SM12x 5090 fallback)](https://github.com/vllm-project/vllm/pull/40899) — **5090 用户必看**

### Qwen 系（Qwen3 → Qwen3-Next → Qwen3.6，**门控线性混合路线**）

#### Qwen3 / Qwen3-Next（前置）

- ⭐⭐ [Qwen3 技术报告详解](https://blog.csdn.net/kebijuelun/article/details/148070438)
- ⭐⭐ [Qwen3 技术报告解读](https://mp.weixin.qq.com/s/4Qz5CB4Upns6rW2jDSJ6gA)

#### Qwen3.6-35B-A3B / Qwen3.6-27B（2026.4.16，必修）

- ⭐⭐⭐ **[SGLang 官方 cookbook · Qwen3.6](https://cookbook.sglang.io/autoregressive/Qwen/Qwen3.6)** — Day-0 部署文档，含 Mamba Radix Cache 等 hybrid 模型特有调度策略（`--mamba-scheduler-strategy`）
- ⭐⭐⭐ **[HuggingFace · Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)** — 模型卡 + 官方部署命令（vLLM ≥ 0.19.0 / SGLang ≥ 0.5.10 / KTransformers）
- ⭐⭐ [博客园 · Qwen3.6-35B-A3B 全面评测](https://www.cnblogs.com/sing1ee/p/19885253)
- ⭐⭐ [Qwen3.5/3.6 混合注意力解析 + 部署](https://dev.tekin.cn/blog/qwen3-5-hybrid-attention-gated-deltanet-moe-deployment)
- ⭐⭐ [Qwen 官方博客](https://qwen.ai/blog?id=qwen3.6-35b-a3b)

### Kimi / MiniMax（2026 双线辨析）

- ⭐⭐⭐ **[LLM 架构新趋势 · Kimi K2 Thinking 和 MiniMax-M2 之后是什么](https://devpress.csdn.net/v1/article/detail/154573876)** — **必读综述**，把"线性 attention vs 全注意力"博弈讲清楚
- ⭐⭐ [Kimi K2 Thinking vs MiniMax M2 全面对比](https://kimi-k2.org/zh/blog/17-kimi-k2-thinking-vs-minimax-m2)
- ⭐⭐ [开源推理模型巅峰对决 · Kimi K2 Thinking vs MiniMax M2](https://blog.gitcode.com/f044616c21b80a6f28db2249be988908.html)

### 论文

- ⭐⭐⭐ Mamba-3: Improved Sequence Modeling using State Space Principles (ICLR 2026 投稿) — 比 Gated DeltaNet 更强，未来 12 个月可能成新热点
- ⭐⭐⭐ Gated Delta Networks: Improving Mamba2 with Delta Rule (arXiv:2412.06464)
