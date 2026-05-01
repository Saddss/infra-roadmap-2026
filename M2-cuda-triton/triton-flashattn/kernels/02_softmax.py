"""
W7 D2：fused softmax

学习点：
    - reduction 在 Triton 中的写法
    - 数值稳定（减最大值）
    - 一个 block 处理一行

TODO W7：填空完成下面 softmax_kernel
"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    out_ptr, in_ptr,
    in_row_stride, out_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    每个 program 处理一行
    """
    row_idx = tl.program_id(0)
    row_start = in_ptr + row_idx * in_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # TODO W7:
    #   1. 加载这一行: x = tl.load(row_start + col_offsets, mask=mask, other=-float("inf"))
    #   2. 减最大值: x_minus_max = x - tl.max(x, axis=0)
    #   3. exp + 求和: numerator = tl.exp(x_minus_max);  denominator = tl.sum(numerator, axis=0)
    #   4. 归一化: out = numerator / denominator
    #   5. 写回 out_ptr + row_idx * out_row_stride + col_offsets

    raise NotImplementedError("W7 任务：完成后删掉这一行")


def softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    out = torch.empty_like(x)
    softmax_kernel[(n_rows,)](out, x, x.stride(0), out.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("需要 GPU"); exit(0)

    x = torch.randn(1823, 781, device="cuda")
    a = softmax(x)
    b = torch.softmax(x, dim=-1)
    print(f"max diff: {(a-b).abs().max().item():.6e}")
