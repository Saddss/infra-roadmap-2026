"""
W2 任务：实现 Multi-Head Attention（torch 版）。

参考资料：
    - "Attention Is All You Need" 论文 Section 3.2
    - karpathy/nanoGPT 中的 CausalSelfAttention 类
    - 苏剑林《缓存与效果的极限拉扯》MHA 部分
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    标准 MHA。每个 head 都有独立的 K/V。

    KV Cache 大小（推理时每 token、每层）:
        2 * num_heads * head_dim * dtype_size
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # TODO W2: 定义 Q/K/V 投影矩阵和输出投影
        # 提示：3 个独立 Linear，或 1 个合并 Linear（更高效）
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,  # (B, T, D)
        mask: torch.Tensor | None = None,  # (T, T) causal mask
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        TODO W2:
            1. 计算 Q, K, V，形状各 (B, T, D)
            2. 把 D 维拆成 (num_heads, head_dim) → (B, num_heads, T, head_dim)
            3. 如果有 kv_cache，把当前 K/V 拼到 cache 末尾
            4. 计算 attention scores: Q @ K.transpose / sqrt(head_dim)
            5. 应用 causal mask
            6. softmax + matmul V → (B, num_heads, T, head_dim)
            7. concat heads → (B, T, D)
            8. out_proj → 最终输出
            9. 返回 (output, new_kv_cache)
        """
        raise NotImplementedError("W2 任务：完成后删掉这一行")

    @staticmethod
    def kv_cache_bytes(seq_len: int, num_heads: int, head_dim: int, dtype_bytes: int = 2) -> int:
        """
        计算单层 MHA 在给定 seq_len 时的 KV Cache 字节数。
        默认 fp16 (2 bytes)。
        """
        return 2 * seq_len * num_heads * head_dim * dtype_bytes
