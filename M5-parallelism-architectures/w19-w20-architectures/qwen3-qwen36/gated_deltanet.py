"""
W20 任务：复现 Gated DeltaNet 状态更新公式（Qwen3.6 / Kimi Linear 用）

核心公式（论文《Gated Delta Networks》）:
    S_t = β_t ⊙ S_{t-1} + Δ_t ⊗ (K_t ⊗ V_t)

其中:
    S_t: 状态矩阵 (head_dim_k × head_dim_v)
    β_t: 门控参数 (head_dim_k × head_dim_k)，对角阵 → 等价于一个 scaling 向量
    Δ_t: 增量参数（对状态的精确更新方向）
    K_t, V_t: 当前 token 的 key 和 value

跑这个脚本:
    python gated_deltanet.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedDeltaNetCell(nn.Module):
    """
    单步 Gated DeltaNet（chunked 版本会更复杂，这里展示核心思想）

    Linear attention 的好处：
        - O(1) 状态内存（与 seq_len 无关）
        - O(L) 时间复杂度
        - 不需要 KV Cache（状态本身就是 cache）
    """

    def __init__(self, dim: int, head_dim_k: int = 64, head_dim_v: int = 128):
        super().__init__()
        self.dim = dim
        self.head_dim_k = head_dim_k
        self.head_dim_v = head_dim_v

        # 一些必要的投影
        self.q_proj = nn.Linear(dim, head_dim_k, bias=False)
        self.k_proj = nn.Linear(dim, head_dim_k, bias=False)
        self.v_proj = nn.Linear(dim, head_dim_v, bias=False)
        self.o_proj = nn.Linear(head_dim_v, dim, bias=False)

        # 门控 β：从输入生成
        self.beta_proj = nn.Linear(dim, head_dim_k, bias=True)

        # 增量 Δ 通过 dt_proj 生成（DeltaNet 风格）
        self.dt_proj = nn.Linear(dim, head_dim_k, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        return: (B, T, D)
        """
        B, T, D = x.shape

        # 状态初始化
        S = torch.zeros(B, self.head_dim_k, self.head_dim_v, device=x.device, dtype=x.dtype)

        # 投影
        q = self.q_proj(x)                                    # (B, T, head_k)
        k = self.k_proj(x)                                    # (B, T, head_k)
        v = self.v_proj(x)                                    # (B, T, head_v)
        beta = torch.sigmoid(self.beta_proj(x))               # (B, T, head_k), 门控值 ∈ (0, 1)
        delta = F.softplus(self.dt_proj(x))                   # (B, T, head_k), 增量幅度 ≥ 0

        outputs = []
        for t in range(T):
            qt = q[:, t]              # (B, head_k)
            kt = k[:, t]              # (B, head_k)
            vt = v[:, t]              # (B, head_v)
            bt = beta[:, t]           # (B, head_k)
            dt = delta[:, t]          # (B, head_k)

            # State update: S = β ⊙ S + Δ ⊗ (k ⊗ v)
            # ⊙: 沿 head_k 维度元素乘（gating S 的"行"）
            # ⊗ (k_t ⊗ v_t): 外积，得到 (head_k, head_v) 的更新矩阵
            kv_outer = torch.einsum("bk,bv->bkv", kt, vt)     # (B, head_k, head_v)
            S = bt.unsqueeze(-1) * S + dt.unsqueeze(-1) * kv_outer

            # 读出: o_t = q_t @ S
            ot = torch.einsum("bk,bkv->bv", qt, S)             # (B, head_v)
            outputs.append(ot)

        out = torch.stack(outputs, dim=1)                      # (B, T, head_v)
        return self.o_proj(out)


def demo():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        print("注意: 在 CPU 上跑")

    cell = GatedDeltaNetCell(dim=128, head_dim_k=32, head_dim_v=64).to(device)

    x = torch.randn(2, 16, 128, device=device)
    y = cell(x)
    print(f"input  shape: {x.shape}")
    print(f"output shape: {y.shape}")
    print()
    print("观察:")
    print("  - 状态 S 的大小是 (head_k=32, head_v=64) = 2048 个元素")
    print("  - 不论 T 多大，状态大小都不变 → 这是线性注意力 O(1) 状态的本质")
    print("  - 与 KV Cache 对比: 普通 attention 在 T=1M 时要缓存 1M*head_dim 个元素")


if __name__ == "__main__":
    demo()
