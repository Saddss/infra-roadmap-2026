"""
W2 任务：可视化 KV Cache 显存占用（MHA vs MQA vs GQA vs MLA）

跑这个脚本，应该输出一张表 + 一张图，作为博客素材。

用法:
    cd ~/sss/infra-roadmap-2026/M1-transformer-fundamentals/mini-llama
    python scripts/compare_kv_cache.py
"""

import matplotlib.pyplot as plt

from mini_llama.attention.mha import MultiHeadAttention
from mini_llama.attention.mqa import MultiQueryAttention
from mini_llama.attention.gqa import GroupedQueryAttention
from mini_llama.attention.mla import MultiHeadLatentAttention


def main():
    # 模拟 LLaMA-3-70B 的配置
    NUM_HEADS = 64
    HEAD_DIM = 128
    DTYPE_BYTES = 2  # fp16

    # MLA 配置参考 DeepSeek-V3
    MLA_D_C = 512
    MLA_HEAD_ROPE = 64

    seq_lens = [1024, 4096, 16384, 65536, 131072]

    print(f"{'seq_len':>10} | {'MHA':>10} | {'MQA':>10} | {'GQA(g=8)':>10} | {'MLA':>10}")
    print("-" * 60)

    mha_data, mqa_data, gqa_data, mla_data = [], [], [], []
    for sl in seq_lens:
        mha = MultiHeadAttention.kv_cache_bytes(sl, NUM_HEADS, HEAD_DIM, DTYPE_BYTES)
        mqa = MultiQueryAttention.kv_cache_bytes(sl, HEAD_DIM, DTYPE_BYTES)
        gqa = GroupedQueryAttention.kv_cache_bytes(sl, num_kv_heads=8, head_dim=HEAD_DIM, dtype_bytes=DTYPE_BYTES)
        mla = MultiHeadLatentAttention.kv_cache_bytes(sl, MLA_D_C, MLA_HEAD_ROPE, DTYPE_BYTES)
        mha_data.append(mha / 1e6); mqa_data.append(mqa / 1e6)
        gqa_data.append(gqa / 1e6); mla_data.append(mla / 1e6)
        print(f"{sl:>10} | {mha/1e6:>9.1f}M | {mqa/1e6:>9.1f}M | {gqa/1e6:>9.1f}M | {mla/1e6:>9.1f}M")

    # 画图
    plt.figure(figsize=(8, 5))
    plt.plot(seq_lens, mha_data, "o-", label="MHA")
    plt.plot(seq_lens, mqa_data, "s-", label="MQA")
    plt.plot(seq_lens, gqa_data, "^-", label="GQA (g=8)")
    plt.plot(seq_lens, mla_data, "d-", label="MLA")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("seq_len"); plt.ylabel("KV Cache per layer (MB)")
    plt.title(f"KV Cache vs seq_len  (num_heads={NUM_HEADS}, head_dim={HEAD_DIM}, fp16)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("kv_cache_comparison.png", dpi=120)
    print("\n图已保存到 kv_cache_comparison.png")


if __name__ == "__main__":
    main()
