"""W1 验收测试：单头 attention（numpy 版）"""

import numpy as np
import pytest

from mini_llama.attention.single_head import scaled_dot_product_attention, softmax


class TestSoftmax:
    def test_sums_to_one(self):
        x = np.random.randn(5, 10)
        out = softmax(x, axis=-1)
        np.testing.assert_allclose(out.sum(axis=-1), np.ones(5), rtol=1e-5)

    def test_numerically_stable(self):
        """大值不应溢出"""
        x = np.array([1000.0, 1001.0, 1002.0])
        out = softmax(x, axis=-1)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))


class TestSingleHeadAttention:
    def test_shape(self):
        Q = np.random.randn(7, 8)
        K = np.random.randn(7, 8)
        V = np.random.randn(7, 16)
        out, attn = scaled_dot_product_attention(Q, K, V)
        assert out.shape == (7, 16)
        assert attn.shape == (7, 7)

    def test_attention_weights_sum_to_one(self):
        Q = np.random.randn(5, 8)
        K = np.random.randn(5, 8)
        V = np.random.randn(5, 8)
        _, attn = scaled_dot_product_attention(Q, K, V)
        np.testing.assert_allclose(attn.sum(axis=-1), np.ones(5), rtol=1e-5)

    def test_causal_mask(self):
        """带 causal mask 后，下三角之外应该是 0"""
        Q = np.random.randn(5, 8)
        K = np.random.randn(5, 8)
        V = np.random.randn(5, 8)
        mask = np.tril(np.ones((5, 5))).astype(bool)
        _, attn = scaled_dot_product_attention(Q, K, V, mask=mask)
        # 上三角应该是 0
        for i in range(5):
            for j in range(i + 1, 5):
                assert abs(attn[i, j]) < 1e-6, f"位置 [{i},{j}] 应为 0，实际 {attn[i,j]}"
