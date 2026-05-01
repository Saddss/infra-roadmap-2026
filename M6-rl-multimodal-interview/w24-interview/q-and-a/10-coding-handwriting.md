# 10 · 手撕代码题

> 字节大模型岗几乎必考。每道题白板能默写 = 真懂。

## Q1: 手撕 GQA（5 分钟内）

#### 题目

实现 GroupedQueryAttention 类，支持任意 dim、num_heads、num_kv_heads。

#### 参考答案

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GQA(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads):
        super().__init__()
        self.dim = dim
        self.h = num_heads
        self.h_kv = num_kv_heads
        self.head_dim = dim // num_heads
        self.group_size = num_heads // num_kv_heads
        
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x, mask=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.h, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.h_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.h_kv, self.head_dim).transpose(1, 2)
        
        # K, V 复制 group_size 次
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)
        
        scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, self.dim)
        return self.o_proj(out)
```

#### 容易写错的点

- `repeat_interleave` 不是 `repeat`
- transpose 顺序：(B, T, h, d) → transpose(1, 2) → (B, h, T, d)
- mask 用 `masked_fill(mask == 0, -inf)` 而不是 `mask * scores`

---

## Q2: 手撕 RMSNorm

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 用 fp32 计算 rms 防止 fp16 下溢
        x_dtype = x.dtype
        x_f = x.float()
        rms = x_f.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x_f / rms).to(x_dtype) * self.weight
```

---

## Q3: 手撕 RoPE

```python
def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64

def apply_rotary_emb(x, freqs_cis):
    # x: (B, T, n_heads, head_dim)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))
    x_rotated = x_complex * freqs_cis
    x_out = torch.view_as_real(x_rotated).flatten(-2)
    return x_out.to(x.dtype)
```

---

## Q4: 手撕 Online Softmax（FlashAttention 核心）

```python
# 给定 q (1, d), 一段段加进来的 k, v 块
m = -float("inf")  # running max
l = 0              # running sum
acc = torch.zeros(d_v)

for k_block, v_block in zip(k_blocks, v_blocks):
    s = q @ k_block.T  # (block_size,)
    m_new = max(m, s.max())
    p = (s - m_new).exp()
    alpha = (m - m_new).exp() if m > -float("inf") else 0
    l = alpha * l + p.sum()
    acc = alpha * acc + p @ v_block
    m = m_new

out = acc / l
```

#### 容易错的点

- 必须先算 m_new 再算 alpha 和 p（顺序很重要）
- alpha 是上一步 m_old 到 m_new 的修正因子
- 最后归一化要除 l（不是单独除每块）

---

## Q5: 手撕 cross-attention（字节三面真题）

```python
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.h = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, 2 * dim)
        self.o_proj = nn.Linear(dim, dim)
    
    def forward(self, decoder_x, encoder_out):
        B, T_d, _ = decoder_x.shape
        _, T_e, _ = encoder_out.shape
        
        q = self.q_proj(decoder_x).view(B, T_d, self.h, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(encoder_out).view(B, T_e, 2, self.h, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        
        scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T_d, -1)
        return self.o_proj(out)
```

---

## Q6: 手撕 GRPO 的 advantage 计算

```python
def compute_grpo_advantage(rewards):
    """
    rewards: (G,) - G 个 response 的奖励
    返回 advantages: (G,)
    """
    mean = rewards.mean()
    std = rewards.std() + 1e-8
    return (rewards - mean) / std
```

---

## Q7: LeetCode 高频（牛客面经里出现过）

字节、抖音大模型岗高频：

- 200. Number of Islands（岛屿数量）
- 33. Search in Rotated Sorted Array
- 415. Add Strings（大数加法）
- 215. Kth Largest Element
- 5. Longest Palindromic Substring
- 102. Binary Tree Level Order Traversal
- 53. Maximum Subarray
- 198. House Robber
- 322. Coin Change
- 300. Longest Increasing Subsequence

每道题至少手写两遍（一遍 brute force + 一遍 optimal）。

---

## 训练计划

W24 周一-周三：每天默写 2-3 题，错了再写一遍。

不是看答案，是默写。看一遍懂了 ≠ 能写。
