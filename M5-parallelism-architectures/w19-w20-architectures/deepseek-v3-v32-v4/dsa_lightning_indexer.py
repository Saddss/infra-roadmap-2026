"""
W19 任务：复现 DSA Lightning Indexer 核心公式

DeepSeek-V3.2-Exp 的 Lightning Indexer 用来从所有历史 KV 中选 top-k=2048 个最相关位置。

公式（V3.2 论文 Equation 1）：
    I_{t,s} = sum_{j=1..H_I} w^I_{t,j} * ReLU(q^I_{t,j} . k^I_s)

其中:
    H_I = indexer head 数（小，比如 64）
    q^I_{t,j} = query 在第 j 个 indexer head 上的向量
    k^I_s = key 在位置 s 的 indexer 向量（所有 head 共享）
    w^I_{t,j} = query-specific 权重，控制每个 head 的重要性

跑这个脚本：
    python dsa_lightning_indexer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightningIndexer(nn.Module):
    """
    DSA Lightning Indexer 简化实现（不带 FP8 量化版）

    输入: x (B, T, D)
    输出: top_k_indices (B, T, K)  — 对每个 query 选出 K 个最相关历史位置
    """

    def __init__(
        self,
        dim: int = 7168,           # V3 hidden
        n_indexer_heads: int = 64, # H_I
        head_dim: int = 128,       # 每个 indexer head 的维度
        rope_dim: int = 64,        # 带 RoPE 的部分
        topk: int = 2048,
    ):
        super().__init__()
        self.n_indexer_heads = n_indexer_heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.topk = topk

        # query 投影: x → q^I (B, T, n_indexer_heads, head_dim)
        # 注意 V3.2 论文中 q^I 也走低秩投影
        self.q_down = nn.Linear(dim, dim // 4, bias=False)
        self.q_up = nn.Linear(dim // 4, n_indexer_heads * head_dim, bias=False)

        # key 投影: x → k^I (B, T, head_dim)，所有 indexer head 共享
        self.k_proj = nn.Linear(dim, head_dim, bias=False)

        # weight 投影: x → w^I (B, T, n_indexer_heads)
        self.w_proj = nn.Linear(dim, n_indexer_heads, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)

        返回 top_k_indices: (B, T, topk) — 每个 query 位置选出的 top-k key 位置
        """
        B, T, D = x.shape

        # query path
        q_c = self.q_down(x)                           # (B, T, D/4)
        q = self.q_up(q_c).view(B, T, self.n_indexer_heads, self.head_dim)  # (B, T, H_I, head_dim)

        # key path（所有 head 共享）
        k = self.k_proj(x)                             # (B, T, head_dim)

        # weight path
        w = self.w_proj(x)                             # (B, T, H_I)

        # 计算 indexer scores: I_{t,s} = sum_j w_{t,j} * ReLU(q_{t,j} . k_s)
        # 形状演变:
        #   q . k: (B, T, H_I, head_dim) x (B, T, head_dim) → (B, T_q, H_I, T_k)
        # 用爱因斯坦求和:
        scores_per_head = torch.einsum("bthd,bsd->bths", q, k)  # (B, T_q, H_I, T_k)
        scores_per_head = F.relu(scores_per_head)
        # 加权求和 over heads
        scores = (scores_per_head * w.unsqueeze(-1)).sum(dim=2)  # (B, T_q, T_k)

        # 因果 mask（query 不能看未来）
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # 取 top-k
        k_use = min(self.topk, T)
        top_k_indices = scores.topk(k_use, dim=-1).indices  # (B, T, k_use)

        return top_k_indices


def demo():
    if not torch.cuda.is_available():
        device = "cpu"; print("注意：在 CPU 上跑会很慢")
    else:
        device = "cuda"

    indexer = LightningIndexer(
        dim=512, n_indexer_heads=8, head_dim=64, topk=128
    ).to(device)

    x = torch.randn(2, 256, 512, device=device)
    indices = indexer(x)
    print(f"输入 shape: {x.shape}")
    print(f"top_k_indices shape: {indices.shape}")
    print(f"每个 query 选出的位置示例（query=10）:")
    print(f"  {indices[0, 10, :10].tolist()}")
    print()
    print("观察：因为 query 10 不能看未来，top_k_indices[0, 10, :] 都应该 ≤ 10")


if __name__ == "__main__":
    demo()
