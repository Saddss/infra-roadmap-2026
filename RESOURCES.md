# 总资源清单

按主题分类的所有学习资料链接。每个阶段的 `resources.md` 是这一份的子集。

---

## 1. Transformer / Attention 基础（M1 用）

### 注意力变体（MHA → MQA → GQA → MLA → DSA → CSA+HCA）

- 📝 苏剑林《缓存与效果的极限拉扯：从 MHA、MQA、GQA 到 MLA》— [科学空间](https://kexue.fm/archives/10091)
- 📝 Yue Shui《Transformer 注意力机制：MHA、MQA 与 GQA 的对比》— [syhya.github.io](https://syhya.github.io/zh/posts/2025-01-16-group-query-attention/)
- 📝 冷眸《Attention 各种变体》— [lengm.cn](https://lengm.cn/post/20250226_attention/)
- 📝 腾讯云《MHA/MQA/GQA/MLA 对比分析》— [cloud.tencent.com](https://cloud.tencent.com/developer/article/2634499)
- 📝 U 深研《国内大模型厂商混合注意力机制》— [unifuncs.com](https://unifuncs.com/s/iCwQrYUv)

### 源码仓库（顺序读）

1. `karpathy/nanoGPT` — 500 行 GPT-2 复现
2. `meta-llama/llama` — `model.py` 中的 GQA 实现
3. `deepseek-ai/DeepSeek-V3` — `model.py` 中的 MLA 实现
4. `deepseek-ai/DeepSeek-V3.2-Exp` — DSA Lightning Indexer
5. `QwenLM/Qwen3` — Gated DeltaNet 混合实现

### 视频（B 站）

- 李宏毅《生成式 AI 导论》Transformer 章节（中文，自带字幕）
- 搜 "Transformer 详解 李沐"
- 搜 "MLA DeepSeek 详解"

---

## 2. GPU 编程：CUDA + Triton（M2 用）

### CUDA

- 📚 教材：**Programming Massively Parallel Processors (PMPP) 第 4 版**
- 📦 中文笔记 + 38 Exercise：[smarterhhsu/PMPP-Learning](https://github.com/smarterhhsu/PMPP-Learning)
- 📝 PMPP 导读：[smarter.xin PMPP 导读](https://smarter.xin/posts/30730973/)
- 📦 国内 CUDA 学习圣地：[BBuf/how-to-optim-algorithm-in-cuda](https://github.com/BBuf/how-to-optim-algorithm-in-cuda)
- 🎥 视频：B 站搜 "CUDA-MODE 课程笔记"（GPU MODE 课程中文版）

### Triton

- 📝 [BBuf · OpenAI Triton 入门笔记三 FusedAttention](https://cloud.tencent.com/developer/article/2392140)
- 📝 [AI-HPC 联盟 · Triton 学习路径](https://ai-hpc.org/guide/08-compiler/triton-learning-path)
- 📝 [SegmentFault · Triton 实现 FlashAttention2](https://segmentfault.com/a/1190000047406258)

### FlashAttention 论文 + 中文图解

- 📝 [图解 FlashAttention V1](https://zhuanlan.zhihu.com/p/669926191)
- 📝 [图解 FlashAttention V2](https://mp.weixin.qq.com/s/5K6yNj23NmNLcAQofHcT4Q)

---

## 3. vLLM / SGLang 推理框架（M3-M4 用）

### vLLM 源码地图

- 📝 [quant67 大模型基础设施工程系列 · PagedAttention 与 Continuous Batching](https://quant67.com/post/llm-infra/12-paged-continuous/12-paged-continuous.html)
- 📝 [Smarter's blog · vLLM 系统拆解 · 七层结构](https://smarter.xin/posts/354d88e4/)
- 📝 [vLLM 底层 PagedAttention 与 Continuous Batching 解释（带源码片段）](https://jishuzhan.net/article/2045034961988812802)
- 📚 vLLM 官方 [Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview.html)

### B 站源码课

- 搜 "vLLM 源码全流程分析"（lyrry1997 系列，带飞书课件）
- 搜 "AI INFRA 学习 LLM 全景图"
- 搜 "vLLM 大模型推理框架 分块显存管理"

### SGLang vs vLLM 横向对比

- 📝 [SGLang vs vLLM 2026 深度对比](https://chenxutan.com/d/1513.html)
- 📝 [网硕互联 · SGLang vs vLLM 核心差异](https://www.wsisp.com/helps/53774.html)

### 工程优化武器库（M4 用）

- 投机采样：搜 "投机采样综述 中文" "Medusa EAGLE MTP"
- 量化：搜 "AWQ 论文解读 中文" "FP8 训练 DeepSeek"
- TritonServer：[CUDA 编程基础与 Triton 模型部署](https://mp.weixin.qq.com/s/mXwJAAyYanmmWqLgK0FZNg)

---

## 4. 分布式并行 + 训练框架（M5 上半 用）

### Megatron 5 大并行

- 📝 [Megatron-LM 深度解析 · 5 大并行策略](https://adg.csdn.net/6952548a5b9f5f31781b8f89.html)
- 📝 [MLTalks · 详解 Megatron Pipeline Parallel](https://www.mltalks.com/posts/3278488319/)
- 📝 [幻方 · 模型并行 Megatron 调优实验](https://www.high-flyer.cn/en/blog/model_parallel-1/inidex/)
- 📝 [Transformer 巨型模型训练核心技术 · Megatron-LM](https://blog.csdn.net/qq_22409661/article/details/145790287)

### Muon 优化器（DeepSeek V4 已采用）

- 搜 "Muon 优化器解读"
- 搜 "Newton-Schulz 迭代 大模型训练"
- 搜 "Kimi Muon 论文"

---

## 5. 2026 最新架构（M5 下半 用 · 必修）

### DeepSeek 系（压缩稀疏全注意力路线）

#### V3 / V3.2-Exp（必修前置）

- 📝 [52nlp · DeepSeek-V3.2-Exp 用稀疏注意力实现高效长上下文](https://www.52nlp.cn/deepseek-v3-2-exp%ef%bc%9a%e7%94%a8%e7%a8%80%e7%96%8f%e6%b3%a8%e6%84%8f%e5%8a%9b%e5%ae%9e%e7%8e%b0%e6%9b%b4%e9%ab%98%e6%95%88%e7%9a%84%e9%95%bf%e4%b8%8a%e4%b8%8b%e6%96%87%e6%8e%a8%e7%90%86)
- 📝 [机器之心 · DSA 公开解读](https://mp.weixin.qq.com/s/WYze9rEZnuZ9l1Y132VJmA)
- 📝 [CSDN · DSA 算法源码分析](https://deepseek.csdn.net/69f0799e54b52172bc7087e5.html)
- 📦 [deepseek-ai/DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)

#### V4（2026.4.24，必修）

- 📝 [量子位 · V4 报告太详尽了 · 484 天换代之路](https://www.163.com/dy/article/KRBULUJ60511DSSR.html)
- 📝 [CSDN · DeepSeek-V4 技术报告全解读 · 从架构到 Infra](https://deepseek.csdn.net/69f040bc0a2f6a37c5a685c2.html)
- 📝 [掘金 · V4 简要解读 · 含详细参数对比表](https://juejin.cn/post/7631898635937497134)
- 📦 [HuggingFace · deepseek-ai/DeepSeek-V4](https://huggingface.co/collections/deepseek-ai/deepseek-v4)

### Qwen / Kimi 系（门控线性混合路线）

#### Qwen3 / Qwen3-Next（前置）

- 📝 [Qwen3 技术报告详解](https://blog.csdn.net/kebijuelun/article/details/148070438)
- 📝 [Qwen3 技术报告解读](https://mp.weixin.qq.com/s/4Qz5CB4Upns6rW2jDSJ6gA)

#### Qwen3.6-35B-A3B（2026.4.16，必修）

- 📝 [博客园 · Qwen3.6-35B-A3B 全面评测](https://www.cnblogs.com/sing1ee/p/19885253)
- 📝 [Qwen3.5/3.6 混合注意力解析 · Gated DeltaNet + MoE 部署](https://dev.tekin.cn/blog/qwen3-5-hybrid-attention-gated-deltanet-moe-deployment)
- 📦 [Qwen 官方博客](https://qwen.ai/blog?id=qwen3.6-35b-a3b)
- 📦 HuggingFace `Qwen/Qwen3.6-35B-A3B`

#### Kimi K2 Thinking & MiniMax M2

- 📝 [LLM 架构新趋势 · Kimi K2 Thinking 和 MiniMax-M2 之后](https://devpress.csdn.net/v1/article/detail/154573876)
- 📝 [开源推理模型巅峰对决 · Kimi K2 Thinking vs MiniMax M2](https://blog.gitcode.com/f044616c21b80a6f28db2249be988908.html)
- 📝 [Kimi K2 Thinking vs MiniMax M2 全面对比](https://kimi-k2.org/zh/blog/17-kimi-k2-thinking-vs-minimax-m2)

---

## 6. RL 训推一体（M6 上半 用）

### R1 复现教程（最佳入门）

- 📝 [Datawhale R1 中文复现教程](https://cloud.tencent.com/developer/article/2494040)（3×A800 跑通 GRPO，420 元）
- 📝 [HuggingFace · Mini-R1 复现 DeepSeek R1 灵光一现](https://hugging-face.cn/blog/open-r1/mini-r1-contdown-game)
- 📝 [Open-R1 · DeepSeek R1 复现项目代码详解](http://152.67.113.27/articles/OpenR1...)

### 框架

- 📦 [OpenRLHF README 中文版](https://github.com/OpenRLHF/OpenRLHF/blob/main/README_zh.md)
- 📦 字节 veRL · `volcengine/verl`
- 📦 HuggingFace TRL · `huggingface/trl`

### 算法解读

- 📝 [柠檬 CC · RLHF 实战系列](https://www.limoncc.com/post/47f9ae3708f426c0/)（PPO/GRPO/REINFORCE++）

---

## 7. ViT / 多模态 / DiT（M6 中段 用）

### Qwen2.5-VL

- 📝 [博客园 · Qwen2.5-VL 技术报告精读](https://www.cnblogs.com/emergence/p/18873748)
- 📝 [Qwen2.5-VL 算法解析](https://jishuzhan.net/article/2045498484711817217)
- 📝 [《Qwen2.5-VL》论文精读笔记](https://www.wsisp.com/helps/55456.html)

### InternVL

- 📝 [InternVL-2.5 训练细节](https://devpress.csdn.net/v1/article/detail/144970962)

### DiT

- 搜知乎 "DiT 详解"

---

## 8. 面试八股（M6 末段 用）

- 📚 [小林 coding · 530+ 大模型面试题](https://xiaolincoding.com/other/ai.html)
- 📝 [字节大模型一面真题](https://blog.csdn.net/2401_84033492/article/details/141093004)
- 📝 [字节大模型一二三面经](https://www.nowcoder.com/feed/main/detail/ef14567a048e4e3fa30cbe28eb5c59e3)
- 📝 [入职字节大模型岗面试分享](https://devpress.csdn.net/v1/article/detail/147386717)
- LeetCode hot 100

---

## 9. 我额外推荐你订阅的内容（每天/每周打开）

### 公众号 / 知乎

- **机器之心**（公众号）：每天看新模型发布
- **量子位**（公众号）：业界动态
- **BBuf 的 CUDA 笔记**：CUDA/Triton 实战
- **Smarter's blog**（[smarter.xin](https://smarter.xin/)）：vLLM/PMPP 系列
- **苏剑林 / 科学空间**（[kexue.fm](https://kexue.fm)）：注意力数学原理

### 推特/X（开 VPN 看）

- @lvwerra（HuggingFace）
- @woosuk_k（vLLM）
- @teknium1
- @karpathy

### GitHub Trending

- 每周看一次 `Trending Python / C++ / CUDA`，找新出的推理优化项目
