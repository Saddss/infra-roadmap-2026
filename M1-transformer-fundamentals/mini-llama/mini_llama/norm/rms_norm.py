"""
W4 任务：实现 RMSNorm（LLaMA 系列用，比 LayerNorm 简单）。

公式：
    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

参考：Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO W4:
            1. 计算 norm: x / sqrt(mean(x^2) + eps)
               提示：用 x.float() 计算（数值稳定），最后转回原 dtype
            2. 乘以 self.weight
            3. 返回
        """
        raise NotImplementedError("W4 任务：完成后删掉这一行")
