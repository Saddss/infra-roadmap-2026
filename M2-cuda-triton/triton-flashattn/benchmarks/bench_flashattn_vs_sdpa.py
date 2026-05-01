"""
W8 必交：benchmark Triton FA2 vs PyTorch SDPA / naive PyTorch

跑出的图直接放在博客里。

用法:
    python benchmarks/bench_flashattn_vs_sdpa.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt

from kernels.flash_attention_v2 import flash_attention_v2  # noqa: E402
# 注意：你也可以直接 import kernels.04_flash_attention_v2 但 04_ 开头不能 import，所以
# 改名建议：写完后把 04_flash_attention_v2.py 复制成 flash_attention_v2.py 即可


def bench(fn, *args, **kwargs):
    """简单的 GPU 计时"""
    for _ in range(3):  # warmup
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    n_iter = 20
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


def naive_attention(q, k, v, causal=True):
    """朴素 PyTorch 实现作为参考"""
    sm_scale = 1.0 / (q.size(-1) ** 0.5)
    scores = q @ k.transpose(-2, -1) * sm_scale
    if causal:
        mask = torch.tril(torch.ones(scores.size(-2), scores.size(-1), device=q.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def main():
    if not torch.cuda.is_available():
        print("需要 GPU"); return

    seq_lens = [256, 512, 1024, 2048, 4096, 8192]
    B, H, D = 1, 16, 64

    triton_times, sdpa_times, naive_times = [], [], []
    for N in seq_lens:
        q = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16)
        k = torch.randn_like(q); v = torch.randn_like(q)

        try:
            t1 = bench(flash_attention_v2, q, k, v, causal=True)
        except NotImplementedError:
            t1 = float("nan")
        t2 = bench(torch.nn.functional.scaled_dot_product_attention, q, k, v, is_causal=True)
        t3 = bench(naive_attention, q, k, v) if N <= 2048 else float("nan")

        triton_times.append(t1); sdpa_times.append(t2); naive_times.append(t3)
        print(f"N={N:>5}: triton {t1:>6.2f} ms | sdpa {t2:>6.2f} ms | naive {t3:>6.2f} ms")

    plt.figure(figsize=(8, 5))
    plt.plot(seq_lens, triton_times, "o-", label="My Triton FA2")
    plt.plot(seq_lens, sdpa_times, "s-", label="PyTorch SDPA")
    plt.plot(seq_lens, naive_times, "^-", label="Naive PyTorch")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("seq_len"); plt.ylabel("Latency (ms)")
    plt.title(f"FlashAttention V2 benchmark (B={B}, H={H}, D={D}, fp16)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("flashattn_benchmark.png", dpi=120)
    print("\n图已保存到 flashattn_benchmark.png")


if __name__ == "__main__":
    main()
