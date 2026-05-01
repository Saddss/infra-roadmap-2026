"""
W3 任务：实现 Multi-head Latent Attention（DeepSeek-V2/V3 的核心创新）。

核心思想：
    1. 把输入压缩成低维 latent vector c，KV Cache 只缓存 c（维度远小于 num_heads * head_dim）
    2. 推理时通过矩阵吸收技巧，让 c 直接参与注意力计算，不用再升维
    3. RoPE 部分单独处理（解耦设计），因为 RoPE 不能和矩阵吸收兼容

参考资料：
    - DeepSeek-V3 论文 Section 2.1
    - 苏剑林《缓存与效果的极限拉扯》MLA 部分
    - 冷眸《Attention 各种变体》MLA 部分
    - DeepSeek-V3 仓库的 model.py 实现

KV Cache 大小（DeepSeek-V3 实测）:
    每 token 仅缓存 (d_c + d_r) ≈ 512 + 64 = 576 维
    而 MHA 需要缓存 2 * num_heads * head_dim = 2 * 128 * 128 = 14336 维
    → MLA 是 MHA 的 1/25
"""

import torch
import torch.nn as nn


class MultiHeadLatentAttention(nn.Module):
    """
    MLA: 通过 latent vector 压缩 KV Cache

    维度配置（参考 DeepSeek-V3）:
        dim = 7168 (hidden)
        num_heads = 128
        head_dim_nope = 128 (no positional encoding 部分)
        head_dim_rope = 64  (rope 部分)
        d_c = 512  (KV latent dim)
        d_c_q = 1536 (Q latent dim, V3 也对 Q 做了低秩投影)

    KV Cache 大小:
        每 token = d_c + head_dim_rope = 512 + 64 = 576 维
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        d_c_kv: int = 512,         # KV 压缩维度
        d_c_q: int = 1536,         # Q 压缩维度
        head_dim_nope: int = 128,  # 不带 RoPE 的部分维度
        head_dim_rope: int = 64,   # 带 RoPE 的部分维度
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.d_c_kv = d_c_kv
        self.d_c_q = d_c_q
        self.head_dim_nope = head_dim_nope
        self.head_dim_rope = head_dim_rope
        self.head_dim = head_dim_nope + head_dim_rope  # 总 head 维度

        # TODO W3: 实现以下投影
        #
        # Q 路径（V3 也对 Q 做了低秩投影）:
        #   q_down: (D → d_c_q)
        #   q_up_nope: (d_c_q → num_heads * head_dim_nope)
        #   q_up_rope: (d_c_q → num_heads * head_dim_rope)
        #
        # KV 路径:
        #   kv_down_nope: (D → d_c_kv)        ← 这个的输出就是要缓存的 c
        #   k_rope:       (D → head_dim_rope) ← shared across all heads, 单独缓存
        #   k_up:         (d_c_kv → num_heads * head_dim_nope)  ← 这个会被"吸收"
        #   v_up:         (d_c_kv → num_heads * head_dim_nope)  ← 这个也会被"吸收"
        #
        # 输出:
        #   out_proj: (num_heads * head_dim_nope → D)
        raise NotImplementedError("W3 任务：定义所有投影后删掉这一行")

    def forward(self, x, freqs_cis=None, kv_cache=None):
        """
        TODO W3 (训练 / 朴素版):
            1. Q 路径:
               q_c = q_down(x)
               q_nope = q_up_nope(q_c).view(B, T, num_heads, head_dim_nope)
               q_rope = q_up_rope(q_c).view(B, T, num_heads, head_dim_rope)
               q_rope = apply_rope(q_rope, freqs_cis)
               q = concat([q_nope, q_rope], dim=-1)

            2. KV 路径:
               c = kv_down_nope(x)                 # (B, T, d_c_kv)  ← 缓存这个
               k_pe = k_rope(x).unsqueeze(2)       # (B, T, 1, head_dim_rope)  ← 也要缓存
               k_pe = apply_rope(k_pe, freqs_cis)
               k_nope = k_up(c).view(B, T, num_heads, head_dim_nope)
               v = v_up(c).view(B, T, num_heads, head_dim_nope)
               k_pe_expanded = k_pe.expand(-1, -1, num_heads, -1)
               k = concat([k_nope, k_pe_expanded], dim=-1)

            3. 标准 attention 计算

        TODO W3 (推理 / 矩阵吸收版，进阶):
            把 q_up_nope 和 k_up 合并：q @ W_qup_nope @ W_kup.T @ c
                                              ↑     这两个矩阵乘以 c 就够了
            把 v_up 和 out_proj 合并：output = (attn @ c) @ (W_vup @ W_out)
            这样不用真的把 c 升维到 num_heads * head_dim_nope，节省计算

            进阶任务：先实现朴素版，跑通后再实现吸收版，对比速度
        """
        raise NotImplementedError("W3 任务：完成后删掉这一行")

    @staticmethod
    def kv_cache_bytes(seq_len: int, d_c_kv: int, head_dim_rope: int, dtype_bytes: int = 2) -> int:
        """每 token 缓存 (d_c_kv + head_dim_rope) 维"""
        return seq_len * (d_c_kv + head_dim_rope) * dtype_bytes
