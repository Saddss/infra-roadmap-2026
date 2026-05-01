# cuda-pmpp · 跟着 PMPP 第 4 版 写 CUDA

每个 chapter 是一个独立的目录，包含：

- `kernel.cu` — 核心 CUDA kernel
- `main.cu` — host 端调用 + 验证 + 计时
- `Makefile` — `make` 编译，`make run` 运行
- `notes.md` — 你做的笔记

## 编译要求

- CUDA Toolkit ≥ 11.0
- nvcc 在 PATH 里
- 一张 NVIDIA GPU（compute capability ≥ 6.1）

```bash
nvcc --version  # 验证
nvidia-smi      # 验证 GPU
```

## 学习路线（W5-W6）

```
ch01-intro             ← 跳过，看书即可（PMPP Ch 1）
ch02-vector-add        ← W5 D1: 入门必做（PMPP Ch 2）
ch03-matmul-naive      ← W5 D2-3: 朴素矩阵乘（PMPP Ch 3）
ch04-tiled-matmul      ← W6 D1-2: Tiling + shared memory（PMPP Ch 5，重点）
ch05-softmax-reduction ← W6 D3-4: 树形规约（PMPP Ch 10）
ch06-bank-conflict     ← W6 D5: 触发并消除 bank conflict（PMPP Ch 6）
```
