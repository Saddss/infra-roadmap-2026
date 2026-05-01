"""
W7 D3-D4：matmul + autotune

学习点：
    - 2D 网格（grid 在 M 和 N 两个维度切）
    - tl.dot（GPU tensor core）
    - autotune 选最佳 BLOCK_M / BLOCK_N / BLOCK_K

TODO W7：填空完成 matmul_kernel
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP_M": 8}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    C = A @ B   shapes: A(M, K) B(K, N) C(M, N)

    TODO W7:
        1. pid → 用 group ordering 提高 L2 cache 命中
        2. 计算这个 block 负责的 (BLOCK_M, BLOCK_N) 区域
        3. 在 K 维度上循环：for k in range(0, K, BLOCK_K):
             加载 A 的 (BLOCK_M, BLOCK_K) tile
             加载 B 的 (BLOCK_K, BLOCK_N) tile
             accumulator += tl.dot(a, b)
        4. 写回 C
    """
    raise NotImplementedError("W7 任务：填完后删掉这一行")


def matmul(a, b):
    M, K = a.shape; K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    matmul_kernel[grid](a, b, c, M, N, K,
                        a.stride(0), a.stride(1),
                        b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1))
    return c


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("需要 GPU"); exit(0)

    M, N, K = 512, 512, 512
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c1 = matmul(a, b)
    c2 = a @ b
    print(f"max diff: {(c1-c2).abs().max().item():.4e}")
