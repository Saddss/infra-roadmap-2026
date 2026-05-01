"""
W4 任务：把 attention + MLP + RMSNorm + RoPE 组装成完整 mini-LLaMA。

参考：
    - karpathy/nanoGPT 的 GPT 类
    - meta-llama/llama 的 model.py
"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from .norm.rms_norm import RMSNorm
from .attention.mha import MultiHeadAttention
from .attention.gqa import GroupedQueryAttention
from .attention.mla import MultiHeadLatentAttention


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 2  # 用于 GQA
    max_seq_len: int = 512
    attention_type: Literal["mha", "gqa", "mla"] = "gqa"


class FeedForward(nn.Module):
    """
    SwiGLU FFN（LLaMA 用）

    TODO W4:
        - hidden_dim ≈ dim * 8/3，向上取整到 multiple_of 倍数
        - 三个 Linear: w1, w3 是门控对，w2 是输出
        - forward: w2(silu(w1(x)) * w3(x))
    """

    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * (dim * 8 / 3) / 2 + 0.5) * 2  # round to even
        # TODO W4: 定义 w1, w2, w3
        raise NotImplementedError("W4 任务：完成后删掉这一行")

    def forward(self, x):
        # TODO W4: silu(w1(x)) * w3(x) → w2
        raise NotImplementedError("W4 任务：完成后删掉这一行")


class TransformerBlock(nn.Module):
    """
    标准 LLaMA block: pre-norm 架构

        h = x + attn(norm(x))
        out = h + ffn(norm(h))

    TODO W4:
        - 根据 config.attention_type 选择 MHA / GQA / MLA
        - 前置 RMSNorm
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # TODO W4: 创建 attention（按 config.attention_type 分支）
        # TODO W4: 创建 FFN
        # TODO W4: 创建两个 RMSNorm
        raise NotImplementedError("W4 任务：完成后删掉这一行")

    def forward(self, x, mask=None, freqs_cis=None):
        raise NotImplementedError("W4 任务：完成后删掉这一行")


class MiniLlama(nn.Module):
    """
    完整 mini-LLaMA。

    TODO W4:
        - tok_embeddings: nn.Embedding(vocab_size, dim)
        - 多个 TransformerBlock 堆叠
        - 最后一层 RMSNorm
        - lm_head: Linear(dim, vocab_size, bias=False)
        - 可选：和 tok_embeddings 共享权重（weight tying）
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # TODO W4: 实现
        raise NotImplementedError("W4 任务：完成后删掉这一行")

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, T)
        return: logits (B, T, vocab_size)

        TODO W4:
            1. tok embeddings
            2. 计算 freqs_cis（cache 在第一次 forward）
            3. 构造 causal mask
            4. 逐层 forward
            5. 最后 norm + lm_head
        """
        raise NotImplementedError("W4 任务：完成后删掉这一行")

    @torch.no_grad()
    def generate(self, prompt_tokens: torch.Tensor, max_new_tokens: int, temperature: float = 1.0):
        """
        最简单的 greedy 或 temperature sampling。
        进阶：支持 KV Cache 加速（W4 stretch goal）。
        """
        raise NotImplementedError("W4 任务（进阶）：可选实现")
