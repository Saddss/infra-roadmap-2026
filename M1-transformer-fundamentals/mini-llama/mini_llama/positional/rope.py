"""
W3 任务：实现 RoPE（Rotary Position Embedding）。

参考资料：
    - 苏剑林《Transformer 升级之路：博采众长的旋转式位置编码》
    - https://kexue.fm/archives/8265
    - LLaMA-2 model.py 中的 apply_rotary_emb
"""

import torch


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    预计算 RoPE 用的旋转角度。

    TODO W3:
        1. 计算 freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        2. t = torch.arange(end)
        3. freqs = torch.outer(t, freqs)  # (end, dim/2)
        4. 用 polar 形式表示复数：freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        5. 返回 freqs_cis  形状 (end, dim/2) complex64
    """
    raise NotImplementedError("W3 任务：完成后删掉这一行")


def apply_rotary_emb(
    x: torch.Tensor,             # (B, T, num_heads, head_dim)
    freqs_cis: torch.Tensor,     # (T, head_dim/2) complex
) -> torch.Tensor:
    """
    应用旋转位置编码。

    TODO W3:
        1. 把 x 后两维 view 成 complex: x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        2. freqs_cis reshape 成可以 broadcast 的形状
        3. x_rotated = x_complex * freqs_cis
        4. 转回 real: torch.view_as_real(x_rotated).flatten(-2)
        5. 返回原 dtype
    """
    raise NotImplementedError("W3 任务：完成后删掉这一行")
