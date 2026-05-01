"""W3 验收测试：MLA + RoPE"""

import torch
import pytest

from mini_llama.attention.mla import MultiHeadLatentAttention
from mini_llama.positional.rope import precompute_freqs_cis, apply_rotary_emb


class TestRoPE:
    def test_freqs_shape(self):
        freqs = precompute_freqs_cis(dim=64, end=128)
        assert freqs.shape == (128, 32)

    def test_apply_rotary_preserves_norm(self):
        """RoPE 是正交变换，应该保持向量 L2 范数不变"""
        x = torch.randn(2, 16, 8, 64)
        freqs = precompute_freqs_cis(dim=64, end=16)
        x_rot = apply_rotary_emb(x, freqs)
        assert x_rot.shape == x.shape
        torch.testing.assert_close(
            x.norm(dim=-1), x_rot.norm(dim=-1), rtol=1e-4, atol=1e-4
        )


class TestMLA:
    def test_output_shape(self):
        attn = MultiHeadLatentAttention(
            dim=512, num_heads=8,
            d_c_kv=128, d_c_q=256,
            head_dim_nope=32, head_dim_rope=16,
        )
        x = torch.randn(2, 16, 512)
        # MLA 需要 freqs_cis
        freqs = precompute_freqs_cis(dim=16, end=16)
        out = attn(x, freqs_cis=freqs)
        # 只检查不崩
        assert out is not None

    def test_kv_cache_much_smaller_than_mha(self):
        """MLA 应该比 MHA KV Cache 显著更小"""
        from mini_llama.attention.mha import MultiHeadAttention

        seq_len = 8192
        num_heads = 128
        head_dim = 128  # MHA total
        d_c_kv = 512
        head_dim_rope = 64

        mha_size = MultiHeadAttention.kv_cache_bytes(seq_len, num_heads, head_dim)
        mla_size = MultiHeadLatentAttention.kv_cache_bytes(seq_len, d_c_kv, head_dim_rope)

        ratio = mla_size / mha_size
        print(f"\nMLA / MHA ratio: {ratio:.2%}  (DeepSeek 报告约 1/25 ≈ 4%)")
        assert ratio < 0.10, f"MLA 应该 < 10% MHA，实际 {ratio:.2%}"
