# M5（W17-W20）：分布式并行 + 2026 最前沿架构

## 上半月（W17-W18）：并行 → 见 `w17-w18-parallelism/`

讲清 TP/PP/DP/EP/CP/SP 的组合策略，跑通 Megatron 4 卡训练。

## 下半月（W19-W20）：新架构 → 见 `w19-w20-architectures/`

吃透 2026 两大旗舰路线的本质分歧：

- **DeepSeek V4 的"压缩稀疏全注意力"**：mHC + CSA/HCA + Muon
- **Qwen3.6 的"门控线性混合注意力"**：Gated DeltaNet 3:1 混合
- 对比辨析：Kimi K2 Thinking、Kimi Linear、MiniMax M2

## 必交产出物（M5 整体）

- [ ] `w17-w18-parallelism/notes/`：6 份并行专题笔记
- [ ] `w17-w18-parallelism/scripts/`：跑通的 Megatron 启动脚本
- [ ] `w19-w20-architectures/arch-comparison-blog/`：综述博客《2026 大模型架构两大流派》
- [ ] `w19-w20-architectures/inference-runs/`：V4-Flash 或 Qwen3.6 跑通笔记
- [ ] DSA Lightning Indexer + Gated DeltaNet 各一段 PyTorch 复现代码

## 这一个月是简历"前沿跟踪能力"的核心证据

讲清这一段，相当于面试官看到："这人确实在跟前沿，不是只会用 vLLM 启动命令的人"。
