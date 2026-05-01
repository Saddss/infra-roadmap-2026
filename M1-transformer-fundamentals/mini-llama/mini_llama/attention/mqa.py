"""
W2 任务：实现 Multi-Query Attention。

核心思想：所有 Q heads 共享同一组 K/V（即 num_kv_heads = 1）
KV Cache 大小是 MHA 的 1/num_heads
"""

import torch
import torch.nn as nn


class MultiQueryAttention(nn.Module):
    """
    MQA: 多个 Q head，共享 1 组 K/V

    KV Cache 大小（推理时每 token、每层）:
        2 * 1 * head_dim * dtype_size  ← 不再乘 num_heads
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # TODO W2:
        #   - q_proj: (D → D)
        #   - k_proj: (D → head_dim)  ← 注意只有 1 个 head
        #   - v_proj: (D → head_dim)
        #   - out_proj: (D → D)
        raise NotImplementedError("W2 任务：定义投影矩阵后删掉这一行")

    def forward(self, x, mask=None, kv_cache=None):
        """
        TODO W2:
            1. Q: (B, T, D) → (B, num_heads, T, head_dim)
            2. K, V: (B, T, head_dim) → (B, 1, T, head_dim)
            3. 在 attention 计算时，K/V 会自动 broadcast 到所有 heads
            4. 其他逻辑同 MHA
        """
        raise NotImplementedError("W2 任务：完成后删掉这一行")

    @staticmethod
    def kv_cache_bytes(seq_len: int, head_dim: int, dtype_bytes: int = 2) -> int:
        return 2 * seq_len * 1 * head_dim * dtype_bytes
