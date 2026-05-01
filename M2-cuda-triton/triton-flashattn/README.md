# triton-flashattn

> Hand-written FlashAttention V2 forward pass in OpenAI Triton, with benchmarks vs PyTorch SDPA.
> Built as a learning project to understand FA's tiling + online softmax tricks.

## Structure

```
kernels/
  01_vector_add.py       # W7 D1: Triton 入门
  02_softmax.py          # W7 D2: fused softmax
  03_matmul.py           # W7 D3-4: matmul + autotune
  04_flash_attention_v2.py  # W8: 主菜
benchmarks/
  bench_softmax.py
  bench_matmul.py
  bench_flashattn_vs_sdpa.py
tests/
  test_*.py              # 正确性验证（vs PyTorch reference）
notes/
  fa_v1_v2_diff.md       # 你的笔记
```

## Setup

```bash
cd ~/sss/infra-roadmap-2026/M2-cuda-triton/triton-flashattn
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 验证 Triton 装好
python -c "import triton; print('triton', triton.__version__)"
```

## Run

```bash
# W7 入门
python kernels/01_vector_add.py
python kernels/02_softmax.py
python kernels/03_matmul.py

# W8 核心
python kernels/04_flash_attention_v2.py
python benchmarks/bench_flashattn_vs_sdpa.py
```

## Roadmap

- [ ] W7 D1: vector add ← 第一个 Triton kernel
- [ ] W7 D2: fused softmax ← 理解 reduction
- [ ] W7 D3-4: matmul + autotune ← 理解 tiling
- [ ] W8 D1-2: 读 FA V1/V2 论文 + 中文图解
- [ ] W8 D3-4: 写 FA V2 forward kernel
- [ ] W8 D5: benchmark vs SDPA + 写博客
