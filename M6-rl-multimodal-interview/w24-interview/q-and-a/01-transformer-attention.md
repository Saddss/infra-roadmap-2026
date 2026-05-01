# 01 · Transformer / Attention 全家桶

> 必考。这一份至少自答 20 题。

## Q1: 解释 self-attention 公式，为什么要除 √d_k？

#### 标准答案

```
attention(Q, K, V) = softmax(QK^T / √d_k) V
```

QK^T 是 q 和 k 的内积，当 d_k 很大时，内积的方差也大（约 d_k 倍），导致 softmax 输入太大、梯度消失。除 √d_k 把方差归一化到约 1。

#### 我的实战例子

```
（W1 在 mini-llama/single_head.py 里实现时，做过去掉 √d_k 的对比实验，
 不除会导致初始 attention 太尖锐 → 训练前 loss 直接爆炸）
```

---

## Q2: MHA / MQA / GQA / MLA 的 KV Cache 大小？

#### 标准答案

每个 token 每层：

```
MHA: 2 * num_heads * head_dim * dtype_size
MQA: 2 * 1         * head_dim * dtype_size  (= MHA / num_heads)
GQA: 2 * num_kv_heads * head_dim * dtype_size  (= MHA / group_size)
MLA: (d_c + d_rope) * dtype_size  ≈ MHA / 25 (DeepSeek-V3)
```

#### 进阶追问

Q: LLaMA-2/3 为什么选 GQA num_kv_heads=8？
A: 8 张卡，每张正好分 1 组 K/V → 计算和带宽利用都最优。

Q: MQA 训练时怎么稳定？
A: MQA 训练不稳定，所以 LLaMA 没用。GQA 在性能和稳定性间折中。

#### 我的实战例子

```
（W2 跑了 compare_kv_cache.py，输出 32K 上下文下：
 MHA = 16 GB, MQA = 0.5 GB, GQA-8 = 2 GB, MLA = 0.6 GB）
```

---

## Q3: MLA 的矩阵吸收技巧是什么？

#### 标准答案

朴素 MLA 推理时：c → 升维 K, V → 计算 attention，每次都要做一次矩阵乘升维。

矩阵吸收：

```
原本: q @ K^T = q @ (W_kup @ c)^T = q @ c^T @ W_kup^T
吸收后: q @ (W_qup_nope @ W_kup^T) @ c
```

把两个矩阵预先乘起来 → c 直接参与 attention，不用真正升维。

→ 推理时 KV Cache 还是只存 c（小），但计算量也不增加。

#### 进阶追问

Q: 为什么 MLA 不能直接和 RoPE 兼容？
A: RoPE 不是线性变换，不能和矩阵吸收一起合并。MLA 解法：把 head_dim 分成 nope（512 维，做矩阵吸收）+ rope（64 维，单独做 RoPE 不吸收）。

#### 我的实战例子

```
（W3 在 mini-llama/mla.py 实现了朴素版和吸收版两个版本，对比 prefill 速度）
```

---

## Q4: DSA（DeepSeek V3.2）和 DSA Lightning Indexer 是什么？

#### 标准答案

DSA = DeepSeek Sparse Attention，在 MLA 上加了：

1. **Lightning Indexer**：用 FP8 算 query 和所有 K 的"分数"
2. **Top-K=2048 选择**：每个 query 只 attend top-k 最相关位置
3. **复杂度从 O(L²) 降到 O(L*k)**

公式：

```
I_{t,s} = sum_j w^I_{t,j} * ReLU(q^I_{t,j} . k^I_s)
```

#### 我的实战例子

```
（W19 在 dsa_lightning_indexer.py 里复现了 50 行 PyTorch 版本，
 跑通 demo 验证了"query 不能看未来"的因果性）
```

---

## Q5: V4 的 CSA 和 V3.2 的 DSA 差别？

#### 标准答案

V4 的 CSA 在 DSA 上多加了一层"压缩"：

```
DSA: 直接对 N 个 token 选 top-k=2048
CSA: 先把 N 压成 N/m (m=4)，再选 top-k
→ N/m * k 的复杂度 < L * k
```

CSA 还配 HCA（更激进压缩 m'=128，全量密集），两者层层交替。

#### 进阶追问

Q: V4 为什么不和 MLA 比，要和 GQA8 比？
A: V4 把 attention 完全重做了，新的 head_dim=512 比 V3 的 128 大，已经不是 MLA 的延续。GQA8 是工业基线（LLaMA-3-70B 选的）。

---

## Q6: Qwen3.6 的 Gated DeltaNet 状态更新公式？

#### 标准答案

```
S_t = β_t ⊙ S_{t-1} + Δ_t ⊗ (K_t ⊗ V_t)

β_t: gate（控制遗忘）
Δ_t: delta（控制更新强度）
S_t: 状态矩阵 (head_dim_k, head_dim_v)
```

特点：
- 状态大小 O(1)（与 seq_len 无关）
- 时间复杂度 O(L)
- 不需要 KV Cache

#### 我的实战例子

```
（W20 在 gated_deltanet.py 复现了，验证了状态大小不论 T 多大都不变）
```

---

## Q7: RoPE 和绝对位置编码的区别？

#### 标准答案

```
绝对位置编码: 直接给 x 加上 pos_embedding[t]
RoPE: 把 q, k 的两个相邻维度看成复数，用 e^(iθ * t) 旋转

效果: 内积 (Rq) · (Rk) 只依赖 t1 - t2（相对位置）
```

优势：
- 长上下文外推更好（YaRN 等扩展技术都基于 RoPE）
- 无需额外参数

---

## Q8: Pre-norm vs Post-norm？

#### 标准答案

```
Post-norm (Transformer 原版):
  out = LayerNorm(x + sublayer(x))

Pre-norm (LLaMA / GPT 主流):
  out = x + sublayer(LayerNorm(x))
```

Pre-norm 更稳，能扩到更深；Post-norm 表达能力略强但难训。

---

## TODO: 补充 12+ 题（自己加）

- Q9: 解释 SwiGLU 和 GeLU 区别
- Q10: KV Cache 的 dtype 选择？
- Q11: ...
（W24 期间逐渐补满到 30 题）
