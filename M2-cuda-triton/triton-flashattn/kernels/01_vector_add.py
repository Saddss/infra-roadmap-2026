"""
W7 D1：第一个 Triton kernel

参考 https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html

学习点：
    - @triton.jit 装饰器
    - tl.program_id(0) 获取当前 program 在 grid 上的 id
    - tl.arange + 指针算 → 加载 / 写回 block
    - tl.constexpr 编译期常量
"""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
    return out


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Triton 需要 GPU。请到一台有 NVIDIA GPU 的机器跑。")
        exit(0)

    torch.manual_seed(0)
    x = torch.rand(98432, device="cuda")
    y = torch.rand(98432, device="cuda")
    out_triton = add(x, y)
    out_torch = x + y
    diff = (out_triton - out_torch).abs().max().item()
    print(f"max diff: {diff:.6e}  ({'PASS' if diff < 1e-5 else 'FAIL'})")
