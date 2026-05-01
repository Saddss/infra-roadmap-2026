# W19-W20：2026 最前沿架构精读

## 目标

吃透 2026 年大模型架构两大流派的本质分歧，能在面试时回答："你怎么看 V4 和 Qwen3.6 的不同路线？"

## 必交产出物

- [ ] **综述博客**《2026 大模型架构两大流派 · 压缩稀疏（V4）vs 门控线性（Qwen3.6）》
  - 至少 5000 字、带 mermaid 演进图、6 个架构对比表
  - 模板见 `arch-comparison-blog/blog-template.md`
- [ ] **跑通笔记**：V4-Flash 或 Qwen3.6-35B-A3B 至少跑通一次推理
  - 笔记见 `inference-runs/`
- [ ] **代码复现**（核心创新点）：
  - DSA Lightning Indexer（约 50-80 行 PyTorch）
  - Gated DeltaNet 状态更新公式（约 50-80 行 PyTorch）

## 每周拆解

### W19：DeepSeek 系（V3 → V3.2 → V4）

- [ ] 读 V3 技术报告 Section 2.1（MLA + DeepSeekMoE）
- [ ] 读 [DSA 解读](https://mp.weixin.qq.com/s/WYze9rEZnuZ9l1Y132VJmA)
- [ ] 读 [V4 技术报告全解读](https://deepseek.csdn.net/69f040bc0a2f6a37c5a685c2.html)
- [ ] 写笔记：`deepseek-v3-v32-v4/notes.md`
- [ ] 复现：DSA Lightning Indexer 50-80 行 PyTorch

### W20：Qwen 系（Qwen3 → Qwen3-Next → Qwen3.6） + Kimi/MiniMax + 写综述

- [ ] 读 Qwen3 技术报告
- [ ] 读 [Qwen3.6 全面评测](https://www.cnblogs.com/sing1ee/p/19885253) + [混合注意力解析](https://dev.tekin.cn/blog/qwen3-5-hybrid-attention-gated-deltanet-moe-deployment)
- [ ] 写笔记：`qwen3-qwen36/notes.md`
- [ ] 复现：Gated DeltaNet 状态更新公式 50-80 行 PyTorch
- [ ] 读 [Kimi K2 Thinking vs MiniMax M2](https://blog.gitcode.com/f044616c21b80a6f28db2249be988908.html)
- [ ] 写笔记：`kimi-minimax/notes.md`
- [ ] **写综述博客并发布**

## 资料

- M5 资源清单详见各子目录下的 notes.md
- 新增：[LLM 架构新趋势 · 综述](https://devpress.csdn.net/v1/article/detail/154573876)
