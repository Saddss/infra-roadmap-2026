# 04 · vLLM / SGLang / TRT-LLM

> 必考。这一份至少自答 25 题。

## Q1: PagedAttention 和 OS 虚拟内存的类比？

#### 标准答案

OS 把进程的逻辑地址空间分页 → 物理内存可以不连续，靠页表映射。

vLLM 把 KV cache 分块（典型 16 token / block）→ 一个 sequence 的 KV 可以散在不同物理位置，靠 block table 映射。

收益：

- 消除内部碎片（不预分配最大长度）
- 不同请求间共享 block（prefix caching）
- 显存利用率从 30-40% 提升到 80%+

---

## Q2: Continuous Batching 是什么？

#### 标准答案

每次 iteration（每生成一个 token）都重新组 batch：

```
传统 static batching: 一车人到齐才发车，最慢的请求决定 batch 速度
continuous batching: 公交车 — 有人下车（结束）马上让新人（新请求）上车
```

核心：在 scheduler.step() 里，每次决定哪些请求进当前 forward。

---

## Q3: vLLM v1 vs v0 主要改了什么？

#### 标准答案

1. 调度器从 Python 搬到更高效的执行层
2. 调度和 forward pipeline（零 CPU 阻塞）
3. 统一 request lifecycle（prefill / decode / preempt / prefix-hit 一条路径）
4. 原生多模态（vision encoder 和 LLM 共享调度）
5. Spec decode、Multi-LoRA、Disaggregated Prefill 都成一等公民

切到 v1 通常免费送 30% 吞吐。

---

## Q4: SGLang 的 RadixAttention 比 vLLM 的 PagedAttention 强在哪？

#### 标准答案

| | vLLM PagedAttention | SGLang RadixAttention |
|---|---|---|
| 粒度 | block 级 hash | token 级前缀树 |
| 命中精度 | 中 | 高 |
| Agent 场景 | 中 | **高（80-95% 命中率）** |

→ Agent / 多轮对话 / 树搜索类场景，SGLang 优势 5-10x。

---

## Q5: chunked prefill 是什么？

#### 标准答案

prefill 阶段（处理 prompt）算力多、延迟低；decode（生成 token）算力少、内存带宽紧。

chunked prefill：把长 prompt 切成多个 chunk，每个 chunk 和正在 decode 的请求一起 forward。

收益：
- 平衡 GPU 利用率
- 避免长 prompt 卡住 decode 流

---

## Q6: vLLM 的 KV Connector 是什么？

#### 标准答案

vLLM v1 的接口，把 KV cache 在不同 worker / 进程 / 机器间传递，为 disaggregated prefill 铺路。

backend：LMCache、Mooncake、NIXL。

---

## Q7: 实际部署时如何选 max_num_batched_tokens？

#### 标准答案

吞吐-延迟 trade-off 的旋钮：

- 大 → throughput ↑，p99 latency ↑（请求等更久）
- 小 → throughput ↓，p99 latency ↓

经验：

```
2K-4K: 适合长 SLA，重视延迟
8K-16K: 适合高吞吐场景
> 16K: 显存压力大，需要权衡
```

#### 我的实战例子

```
（W11 跑了 bench_vllm_throughput_latency.py，发现在我的机器上
 4096 是 throughput-latency 拐点）
```

---

## Q8: Speculative Decoding 怎么工作？

#### 标准答案

```
Draft model（小，快）一次 forward 出 k 个候选 token
Target model（大，慢）一次 forward 验证这 k 个 token
  - 如果都接受 → 一次 forward 出 k+1 个 token（赚到）
  - 如果第 i 个拒绝 → 接受前 i 个 token + 用 target 给的第 i+1 个

acceptance rate 越高，越省钱
```

变体：
- Medusa：去掉 draft model，用多个并行 head 出 k 个候选
- EAGLE：用 hidden state 而非 token 做 draft
- MTP：DeepSeek-V3 训练时学习多 token 预测（不需要 draft model）

---

## Q9: AWQ vs GPTQ 区别？

#### 标准答案

```
GPTQ: 逐层贪心量化，重建误差最小化
AWQ: 保护"显著"权重通道（不量化或低 bit），用 calibration set 找显著通道
```

AWQ 一般精度更好，GPTQ 校准更快。

---

## Q10: vLLM 在 RLHF 训练中的角色？

#### 标准答案

PPO / GRPO 训练的瓶颈是 rollout（actor 模型生成 response）。OpenRLHF / veRL 让 actor 模型用 vLLM 做 rollout：

```
传统: HF .generate() → 慢
新解法: vLLM engine + actor 共享权重（hybrid engine）→ 10x 加速
```

→ 这就是"训推一体"，是当下高溢价方向。

---

## TODO: 补充 15 题（W24 期间）
