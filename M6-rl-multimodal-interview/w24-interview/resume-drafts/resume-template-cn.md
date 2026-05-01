# 简历模板（中文版）

> 字节、阿里、月之暗面、DeepSeek 投递专用

---

## <你的名字>

📧 your.email@example.com  ｜  📱 +86-xxx-xxxx-xxxx  ｜  GitHub: github.com/<你>  ｜  知乎: zhihu.com/people/<你>

求职意向：**大模型推理 / 训练 Infra 工程师**（高级 / 资深）

---

## 教育背景

**<学校>** — <学位> · <专业> · <year - year>

GPA: __ / 4.0  ｜  核心课程：操作系统、计算机网络、分布式系统、深度学习、NLP

---

## 工作经验

**<现公司> — <职位> · <year> - 至今**

- 用一句话讲你做的核心业务
- 用一句话讲你的技术贡献（量化）
- 用一句话讲你的协作影响

---

## 核心技术能力

**大模型推理框架**：vLLM（v1 源码精读 + X 个 PR 已合并）、SGLang、TRT-LLM、TritonServer
**GPU 编程**：CUDA C++ (PMPP 基础)、OpenAI Triton（独立实现 FlashAttention V2）
**分布式训练**：Megatron-LM 5 大并行（TP/PP/DP/EP/CP/SP）、FSDP、DeepSpeed ZeRO
**强化学习**：PPO / DPO / GRPO / REINFORCE++（OpenRLHF 跑通 R1 类复现）
**新架构跟踪**：精读 DeepSeek V3/V3.2/V4 + Qwen3/Qwen3.6 + Kimi/MiniMax
**编程语言**：Python（5+ 年）、C++、CUDA、Triton

---

## 开源贡献

- **vLLM (vllm-project/vllm)**：提交 X 个 PR，Y 个已合并
  - <PR #1234>: <一句话描述改了什么>
  - <PR #5678>: <一句话描述改了什么>
- **SGLang / OpenRLHF / 其他**：<如果有的话>

---

## 个人项目

### mini-llama · 从零实现 LLaMA 注意力变体（2026.05）

`Python · PyTorch · NumPy`

- 独立实现 MHA / MQA / GQA / MLA 四种注意力 + RoPE + RMSNorm，组装成完整 mini-LLaMA
- 在 tinyshakespeare 上跑通训练，验证三种 attention 切换效果一致
- 输出 KV Cache 显存对比实验：32K 上下文下 MLA 比 MHA 减少 92% 显存
- GitHub: <repo 链接>  ｜ 配套博客: <知乎链接>

### triton-flashattn · 用 Triton 实现 FlashAttention V2 forward（2026.06）

`Python · OpenAI Triton`

- 独立实现 100 行 Triton kernel 完成 FlashAttention V2 forward pass
- 在 8K seq_len 下吞吐为 PyTorch SDPA 的 X×（fp16）
- GitHub: <repo 链接>  ｜ 配套博客: <知乎链接>

### vLLM 源码精读 + 开源贡献（2026.07-08）

`vLLM · Python · C++/CUDA`

- 通读 vLLM v1 引擎、调度器、PagedAttention 三大核心模块（X 万行代码）
- 向主仓库提交 X 个 PR（Y 个已合并）：<列 1-2 个最重要的>
- 输出 3 篇 5000+ 字技术博客，发布到知乎累计 X 赞 / Y 评论

### 推理框架横向 Benchmark 报告（2026.09）

`vLLM · SGLang · TRT-LLM · 量化`

- 完成 vLLM / SGLang / TRT-LLM 三大主流框架端到端 benchmark 对比
- 覆盖 throughput / TTFT / 量化（AWQ/GPTQ/FP8）/ spec decoding / prefix caching 5 大维度
- 在 Agent 多轮对话场景下证明 SGLang 比 vLLM 提升 X%
- 报告链接: <知乎>

### 2026 大模型架构两大流派综述（2026.10）

`研究 · 跟踪前沿`

- 精读 DeepSeek V4（CSA+HCA+mHC+Muon）和 Qwen3.6（Gated DeltaNet 3:1 混合）技术报告
- 复现两个核心创新：DSA Lightning Indexer + Gated DeltaNet（各 50-80 行 PyTorch）
- 跑通 V4-Flash / Qwen3.6 推理，记录性能数据
- 综述博客 X 字，知乎 X 赞

---

## 实习 / 项目经验（如有）

<同上格式>

---

## 其他

- 知乎专栏 / 公众号: <链接，如果你写了 8-10 篇博客就肯定有专栏>
- 中英双语：英文流利（IELTS X / TOEFL X）

---

## 简历检查清单（投出去前对着勾）

- [ ] 每个项目都有一个量化数据
- [ ] 每个项目都有 GitHub 链接 + 博客链接
- [ ] 没有"熟悉" "了解" "学习过" 这种虚词，只有"实现" "完成" "提升 X%"
- [ ] 一页 A4 能放下（中文 1 页，英文 1 页）
- [ ] 没有错别字（让朋友看一遍）
- [ ] 字体一致（推荐：思源黑体 / Helvetica）
- [ ] 联系方式正确（特别是邮箱）
