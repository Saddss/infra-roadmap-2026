# M2（W5-W8）：CUDA + Triton 实战

## 目标

能在 Triton 里独立写出 FlashAttention V2 forward，并解释它每一步在做什么、为什么比 PyTorch SDPA 快。

## 必交产出物

- [ ] `triton-flashattn/` 仓库（已搭好脚手架）
- [ ] benchmark 脚本输出 throughput vs `F.scaled_dot_product_attention` 对比图
- [ ] 博客一篇《我用 Triton 100 行写了一个 FlashAttention，并解释它为什么快》

## 每周拆解

### W5：CUDA 基础（PMPP 第 1-4 章）

- [ ] 读 PMPP 1-4 章 + 跟着 [smarterhhsu/PMPP-Learning](https://github.com/smarterhhsu/PMPP-Learning) 的中文笔记对照
- [ ] 写 `cuda-pmpp/ch02-vector-add/`：vector add（理解 thread/block/grid）
- [ ] 写 `cuda-pmpp/ch03-matmul-naive/`：朴素 matmul
- [ ] 看 B 站 "CUDA-MODE 课程笔记 Lecture 1-3"

### W6：CUDA Tiling + 共享内存（PMPP 第 5-6 章）

- [ ] 读 PMPP 5-6 章
- [ ] 写 `cuda-pmpp/ch04-tiled-matmul/`：tiled matmul，对比朴素版速度
- [ ] 写 `cuda-pmpp/ch05-softmax-reduction/`：tree-based reduction
- [ ] 写 `cuda-pmpp/ch06-bank-conflict/`：触发并消除 bank conflict

### W7：Triton 三大入门教程

- [ ] 跑通 OpenAI Triton 官方教程 1-3（vector add → softmax → matmul）
- [ ] 写 `triton-flashattn/kernels/01_vector_add.py`、`02_softmax.py`、`03_matmul.py`
- [ ] 理解 `tl.program_id`、`tl.make_block_ptr`、`tl.dot`、`tl.constexpr`
- [ ] 读 BBuf《Triton 入门笔记三 FusedAttention》

### W8：Triton 实现 FlashAttention V2 forward + benchmark

- [ ] 读 FlashAttention V1/V2 论文 + 中文图解（资料链接见 resources.md）
- [ ] 写 `triton-flashattn/kernels/04_flash_attention_v2.py`
- [ ] 写 `benchmarks/bench_flashattn_vs_sdpa.py`，输出对比图
- [ ] 写博客发出去

## 阶段切换检查表（进 M3 前）

- [ ] `triton-flashattn/` 已 push 到 GitHub
- [ ] benchmark 图证明 Triton 实现比 naive PyTorch 快（即使比 SDPA 慢也没关系）
- [ ] 博客发布
- [ ] `weekly-tracker.md` 4 周记录都填了

## 资料

→ 见 [`resources.md`](./resources.md)

## 周记

→ 见 [`weekly-tracker.md`](./weekly-tracker.md)
