"""
W8：FlashAttention V2 forward 的 Triton 实现 —— M2 的核心产出

参考实现（你可以先抄，再逐行理解，最后凭记忆重写）：
    - https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py
    - BBuf《OpenAI Triton 入门笔记三 FusedAttention》

要理解的核心点：
    1. 为什么 V2 把 Q 移到外循环、K/V 移到内循环（V1 反过来）
    2. Online softmax 的 m_i / l_i 更新公式：
         m_new = max(m_old, m_block)
         l_new = exp(m_old - m_new) * l_old + sum(exp(qk - m_new))
         acc *= exp(m_old - m_new)  ← 修正之前累加的输出
         acc += exp(qk - m_new) @ V
    3. 为什么 V2 在 seq_len 维度上启动 grid（提高 SM 占用率）

完成验收：
    - 跑 `python kernels/04_flash_attention_v2.py` 输出 max diff < 1e-2
    - 跑 `python benchmarks/bench_flashattn_vs_sdpa.py` 输出对比图
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, Out, M,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """
    TODO W8: 实现这个 kernel
    
    grid: (cdiv(N_CTX, BLOCK_M), Z * H, 1)
    每个 program 处理一个 (BLOCK_M, HEAD_DIM) 的 Q 块

    伪代码:
        start_m = program_id(0)
        bh = program_id(1)  # batch * head
        z = bh // H; h = bh % H

        加载 Q 这个 block (BLOCK_M, HEAD_DIM)
        m_i = -inf 数组 (BLOCK_M,)
        l_i = 0 数组   (BLOCK_M,)
        acc = 0 数组   (BLOCK_M, HEAD_DIM)

        for start_n in 0..N_CTX step BLOCK_N:
            如果 CAUSAL 且 start_n > start_m * BLOCK_M, break
            加载 K, V 这一块 (BLOCK_N, HEAD_DIM)
            qk = Q @ K.T * sm_scale  (BLOCK_M, BLOCK_N)
            如果 CAUSAL: 应用 mask (qk = qk if i >= j else -inf)
            m_ij = max(qk, axis=1)
            m_new = max(m_i, m_ij)
            p = exp(qk - m_new[:, None])
            alpha = exp(m_i - m_new)
            l_new = alpha * l_i + sum(p, axis=1)
            acc = alpha[:, None] * acc + p @ V
            m_i = m_new
            l_i = l_new

        acc /= l_i[:, None]
        写回 Out
    """
    raise NotImplementedError("W8 任务：实现 FA2 forward kernel")


def flash_attention_v2(q, k, v, causal=False, sm_scale=None):
    """
    q, k, v: (B, H, N, D)
    """
    B, H, N, D = q.shape
    assert D in {16, 32, 64, 128}, "head_dim 必须是 2 的幂且 ≤ 128"
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    BLOCK_M, BLOCK_N = 128, 64
    out = torch.empty_like(q)
    M = torch.empty((B, H, N), device=q.device, dtype=torch.float32)

    grid = (triton.cdiv(N, BLOCK_M), B * H, 1)
    _flash_attn_fwd_kernel[grid](
        q, k, v, out, M, sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D, CAUSAL=causal,
    )
    return out


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("需要 GPU"); exit(0)

    # 正确性检查 vs PyTorch reference
    B, H, N, D = 1, 4, 1024, 64
    q = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16)
    k = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16)
    v = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16)

    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = flash_attention_v2(q, k, v, causal=True)
    diff = (ref - out).abs().max().item()
    print(f"max diff vs SDPA: {diff:.4e}  ({'PASS' if diff < 1e-2 else 'FAIL'})")
