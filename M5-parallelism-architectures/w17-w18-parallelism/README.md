# W17-W18：分布式并行

## 目标

讲清 5 大并行策略，能在白板上画出每种并行的通信模式，能设计 70B / 405B 模型的 3D 并行配置。

## 必交笔记

- [ ] `notes/01-data-parallel.md` — DP / DDP / FSDP / DeepSpeed ZeRO-1/2/3 区别
- [ ] `notes/02-tensor-parallel.md` — TP 怎么切矩阵（行并行 vs 列并行）
- [ ] `notes/03-pipeline-parallel.md` — GPipe / PipeDream / 1F1B / Interleaved
- [ ] `notes/04-context-parallel.md` — CP 长序列处理
- [ ] `notes/05-expert-parallel.md` — EP 给 MoE
- [ ] `notes/06-sequence-parallel.md` — SP 减激活内存
- [ ] `notes/07-muon-optimizer.md` — Muon 优化器（V4 用）

## 必交脚本

- [ ] `scripts/run_megatron_4gpu.sh` — 跑通的 4 卡 Megatron GPT 训练
- [ ] `scripts/compare_zero_levels.sh` — DeepSpeed ZeRO-1/2/3 切换实验

## 每周拆解

### W17：DP / TP / PP 主线

- [ ] 看视频："Megatron-LM 5 大并行策略"（B 站）
- [ ] 读：[Megatron-LM 深度解析](https://adg.csdn.net/6952548a5b9f5f31781b8f89.html)
- [ ] clone `NVIDIA/Megatron-LM`，按 README 跑通 4 卡 GPT-2 训练（云租 4×A10/A100）
- [ ] 实验：分别跑 (TP=1, PP=4)、(TP=2, PP=2)、(TP=4, PP=1)，记录 throughput
- [ ] 写笔记 01-03

### W18：CP / EP / SP + Muon

- [ ] 读：[MLTalks 详解 Pipeline Parallel](https://www.mltalks.com/posts/3278488319/)
- [ ] 读：[幻方 · 模型并行 Megatron 调优实验](https://www.high-flyer.cn/en/blog/model_parallel-1/inidex/)
- [ ] 理解 Sequence Parallel 在 Megatron 里的实现（`--sequence-parallel`）
- [ ] 理解 EP（专家并行）+ TP 必须开 SP 的原因
- [ ] 看 Muon 优化器原理：搜 "Muon Newton-Schulz 迭代"
- [ ] 写笔记 04-07

## 资料

- [Megatron-LM 深度解析 5 大并行](https://adg.csdn.net/6952548a5b9f5f31781b8f89.html)
- [MLTalks · 详解 Megatron Pipeline Parallel](https://www.mltalks.com/posts/3278488319/)
- [幻方 · 模型并行 Megatron 调优实验](https://www.high-flyer.cn/en/blog/model_parallel-1/inidex/)
- [Transformer 巨型模型训练核心技术 · Megatron-LM](https://blog.csdn.net/qq_22409661/article/details/145790287)
- DeepSpeed ZeRO 论文
- FSDP 官方文档
