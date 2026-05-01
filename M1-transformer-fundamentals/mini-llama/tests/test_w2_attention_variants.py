"""W2 验收测试：MHA / MQA / GQA"""

import torch
import pytest

from mini_llama.attention.mha import MultiHeadAttention
from mini_llama.attention.mqa import MultiQueryAttention
from mini_llama.attention.gqa import GroupedQueryAttention


class TestMHA:
    def test_output_shape(self):
        attn = MultiHeadAttention(dim=128, num_heads=8)
        x = torch.randn(2, 16, 128)
        out, _ = attn(x)
        assert out.shape == (2, 16, 128)

    def test_kv_cache_size(self):
        # MHA: 2 * num_heads * head_dim * dtype_bytes
        size = MultiHeadAttention.kv_cache_bytes(seq_len=1024, num_heads=8, head_dim=64, dtype_bytes=2)
        assert size == 2 * 1024 * 8 * 64 * 2  # = 2,097,152 bytes = 2 MB


class TestMQA:
    def test_output_shape(self):
        attn = MultiQueryAttention(dim=128, num_heads=8)
        x = torch.randn(2, 16, 128)
        out, _ = attn(x)
        assert out.shape == (2, 16, 128)

    def test_kv_cache_smaller_than_mha(self):
        mha_size = MultiHeadAttention.kv_cache_bytes(1024, 8, 64)
        mqa_size = MultiQueryAttention.kv_cache_bytes(1024, 64)
        assert mqa_size == mha_size // 8  # MQA 是 MHA 的 1/num_heads


class TestGQA:
    def test_output_shape(self):
        attn = GroupedQueryAttention(dim=128, num_heads=8, num_kv_heads=2)
        x = torch.randn(2, 16, 128)
        out, _ = attn(x)
        assert out.shape == (2, 16, 128)

    def test_degenerates_to_mha_when_kvheads_equals_heads(self):
        """num_kv_heads = num_heads 时，KV Cache 等于 MHA"""
        gqa_size = GroupedQueryAttention.kv_cache_bytes(1024, num_kv_heads=8, head_dim=64)
        mha_size = MultiHeadAttention.kv_cache_bytes(1024, num_heads=8, head_dim=64)
        assert gqa_size == mha_size

    def test_degenerates_to_mqa_when_kvheads_is_one(self):
        """num_kv_heads = 1 时，KV Cache 等于 MQA"""
        gqa_size = GroupedQueryAttention.kv_cache_bytes(1024, num_kv_heads=1, head_dim=64)
        mqa_size = MultiQueryAttention.kv_cache_bytes(1024, head_dim=64)
        assert gqa_size == mqa_size


class TestKVCacheComparison:
    """W2 必须输出的对比表 —— 跑这个测试就能看到三种方案的差异"""

    def test_print_comparison_table(self):
        seq_len = 32 * 1024  # 32K context
        num_heads = 32
        head_dim = 128

        mha = MultiHeadAttention.kv_cache_bytes(seq_len, num_heads, head_dim)
        mqa = MultiQueryAttention.kv_cache_bytes(seq_len, head_dim)
        gqa_8 = GroupedQueryAttention.kv_cache_bytes(seq_len, num_kv_heads=8, head_dim=head_dim)

        def mb(x): return x / 1024 / 1024

        print(f"\n=== KV Cache @ seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim} ===")
        print(f"  MHA       : {mb(mha):>8.2f} MB")
        print(f"  MQA       : {mb(mqa):>8.2f} MB  ({100*mqa/mha:.1f}% of MHA)")
        print(f"  GQA(g=8)  : {mb(gqa_8):>8.2f} MB  ({100*gqa_8/mha:.1f}% of MHA)")

        assert mqa < gqa_8 < mha
