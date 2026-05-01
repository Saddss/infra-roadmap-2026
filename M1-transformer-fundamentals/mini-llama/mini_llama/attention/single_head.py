"""
W1 任务：用 numpy 实现单头 scaled dot-product attention。

不要用 torch！纯 numpy 写，强迫自己理解每一步。
"""

import numpy as np


def scaled_dot_product_attention(
    Q: np.ndarray,  # (seq_len, d_k)
    K: np.ndarray,  # (seq_len, d_k)
    V: np.ndarray,  # (seq_len, d_v)
    mask: np.ndarray | None = None,  # (seq_len, seq_len), True 表示保留
) -> np.ndarray:
    """
    计算 Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    TODO W1:
        1. 计算 scores = Q @ K.T
        2. 缩放 scores /= sqrt(d_k)  —— 思考：为什么要除 sqrt(d_k)？写在注释里
        3. 如果有 mask，把 mask=False 的位置设为 -inf
        4. 在最后一个维度做 softmax 得到 attention weights
        5. 用 weights @ V 得到输出
        6. 返回 (output, attention_weights)

    返回:
        output: (seq_len, d_v)
        attn:   (seq_len, seq_len)  注意力权重，用于可视化
    """
    raise NotImplementedError("W1 任务：完成后删掉这一行")


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    数值稳定的 softmax（要减去最大值再 exp）。

    TODO W1: 自己实现一个数值稳定的 softmax，不准用 scipy.special.softmax。
    """
    raise NotImplementedError("W1 任务：完成后删掉这一行")


# === 验证你的实现 ===
if __name__ == "__main__":
    np.random.seed(42)
    Q = np.random.randn(5, 8)
    K = np.random.randn(5, 8)
    V = np.random.randn(5, 8)
    output, attn = scaled_dot_product_attention(Q, K, V)
    print("output.shape:", output.shape)
    print("attn.shape:", attn.shape)
    print("attn 每行求和应近似 1:", attn.sum(axis=-1))
