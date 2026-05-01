# 07 · Muon 优化器（DeepSeek V4 用，2026 新八股）

## 核心思想

把矩阵参数的更新方向"正交化"再用：

```
普通 SGD:    θ = θ - lr * g
普通 Adam:   θ = θ - lr * m / (sqrt(v) + eps)
Muon:        θ = θ - lr * orthogonalize(m)
```

`orthogonalize(M)` 把矩阵 M 投影到最近的正交矩阵 → 用 Newton-Schulz 迭代近似。

## 为什么有效

- 矩阵参数（attention QKV、FFN）有内在的"低秩 + 正交"结构
- Adam 的对角化更新破坏了这种结构
- Muon 保留了矩阵的几何性质，收敛更快、更稳定

## DeepSeek V4 的实现

```
hybrid Newton-Schulz 迭代：10 步分两段
- 前 5 步快速收敛
- 后 5 步精修
```

V4 把 AdamW 替换为 Muon 在大部分参数上 → 收敛更快、loss 曲线更平滑。

**首个在大规模 MoE 上成功应用 Muon 的工作**。

## 来源

Muon 优化器最早由 Kimi 团队提出（Moonlight），DeepSeek V4 沿用并改造。

## 实现要点（伪代码）

```python
def newton_schulz(M, n_iter=5):
    """近似 (M @ M.T)^(-1/2) @ M = orthogonalize(M)"""
    M = M / M.norm()  # 归一化
    for _ in range(n_iter):
        A = M @ M.T
        I = identity_like(A)
        M = (3 * M - A @ M) / 2  # ≈ M @ (3I - A) / 2
    return M

def muon_step(theta, grad, lr, momentum_buffer, beta=0.95):
    momentum_buffer = beta * momentum_buffer + grad
    update = newton_schulz(momentum_buffer, n_iter=5)
    theta = theta - lr * update
    return theta, momentum_buffer
```

## 实验记录

```
（如果 W18 想动手验证，可以在 mini-llama 训练里把 AdamW 替换成 Muon，对比收敛速度）
```

## 面试可讲

```
（Muon vs AdamW 区别？为什么 Muon 适合矩阵参数不适合标量？）
```
