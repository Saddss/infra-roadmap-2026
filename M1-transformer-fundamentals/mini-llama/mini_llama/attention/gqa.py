"""
W2 任务：实现 Grouped-Query Attention（LLaMA-2/3 用的方案）。

核心思想：把 num_heads 分成 num_groups 组，每组共享一组 K/V
- num_kv_heads = num_groups
- 当 num_groups = num_heads → 退化为 MHA
- 当 num_groups = 1 → 退化为 MQA
LLaMA-2/3 选 num_groups = 8（每张 GPU 1 组）

参考实现：
    - meta-llama/llama 的 model.py
    - 苏剑林《缓存与效果的极限拉扯》GQA 部分
"""

import torch
import torch.nn as nn


class GroupedQueryAttention(nn.Module):
    """
    GQA: num_heads 个 Q head 分成 num_kv_heads 组

    KV Cache 大小:
        2 * num_kv_heads * head_dim * dtype_size
    """

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        assert num_heads % num_kv_heads == 0, "num_heads 必须是 num_kv_heads 的倍数"

        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.group_size = num_heads // num_kv_heads  # 每组多少 Q head

        # TODO W2:
        #   - q_proj: (D → num_heads * head_dim)        = (D → D)
        #   - k_proj: (D → num_kv_heads * head_dim)
        #   - v_proj: (D → num_kv_heads * head_dim)
        #   - out_proj: (D → D)
        raise NotImplementedError("W2 任务：定义投影后删掉这一行")

    def forward(self, x, mask=None, kv_cache=None):
        """
        TODO W2:
            1. Q: (B, T, D) → (B, num_heads, T, head_dim)
            2. K, V: (B, T, num_kv_heads * head_dim) → (B, num_kv_heads, T, head_dim)
            3. 把 K, V repeat_interleave group_size 次 → (B, num_heads, T, head_dim)
               提示：用 torch.repeat_interleave 或 einops.repeat
            4. 后续同 MHA
        """
        raise NotImplementedError("W2 任务：完成后删掉这一行")

    @staticmethod
    def kv_cache_bytes(seq_len: int, num_kv_heads: int, head_dim: int, dtype_bytes: int = 2) -> int:
        return 2 * seq_len * num_kv_heads * head_dim * dtype_bytes
