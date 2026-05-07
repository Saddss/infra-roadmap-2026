# W16 进阶 · 推理系统分离架构全谱系

> 这一份是 W16 浅入门 [`notes.md`](./notes.md) 的进阶版。
> 涵盖 PD 分离、AF 分离、大 EP / EPLB、xPyD 弹性 PD、FlashInfer 等 2025-2026 推理 infra 最热的系统级方向。
> **如果你 W16 时间紧，先看 [`notes.md`](./notes.md) 浅版；W24 之前再回来读这一份**。

> 收录原则见顶层 [RESOURCES.md](../../RESOURCES.md) 的"收录与排序原则"。

## 0. 一张图看清"分离"的演进逻辑

```
单 GPU 单 model 单 forward  (vLLM v0 早期)
        ↓
PD 分离: prefill / decode 切到不同 GPU 池  (Mooncake 2024)
        ↓
xPyD: x 个 prefill + y 个 decode 弹性比例  (vLLM PR #18242, SGLang #9442)
        ↓
EPDG 分离: encoder/prefill/decode/generate 四级  (vLLM-Omni 2025.11)
        ↓
AF 分离: attention / FFN 算子级分离  (MegaScale-Infer 2025.4)
```

每一步分离都是因为：**前一代的"耦合"成为了某个维度的瓶颈**。

## 1. PD 分离（Prefill/Decode Disaggregation）

### 1.1 为什么需要

prefill 和 decode 的**根本矛盾**：

| | Prefill | Decode |
|---|---|---|
| 计算特性 | compute-bound | memory-bound |
| Batch 维度 | seq_len 大 | batch_size 大 |
| GPU 利用率 | 容易跑满 SM | 容易跑满 HBM 带宽 |
| 关键 SLO | TTFT (Time to First Token) | TBT (Time Between Tokens) |

放同一卡上 → 互相干扰（decode 被 prefill 卡住，TBT 抖动）

### 1.2 Mooncake（**月之暗面 + 清华，工业界标杆**）

- 论文一作章明星 = 清华助理教授 + 月之暗面 KVCache.AI 团队负责人
- 核心：**KVCache-centric 调度** —— PD 之外还独立出一个 KVCache 池（用全集群 CPU/DRAM/SSD）
- 性能：相比 vLLM baseline 在某些场景吞吐 +525%

📚 已收录在顶层 [RESOURCES.md 第 4 节](../../RESOURCES.md) 和 [`M4 resources.md`](../resources.md)

### 1.3 LMCache（**学术界标杆 + 多级 KV cache 工程实现**）

- 由 LMCache.ai 维护，开源
- 三级架构：GPU HBM ←→ CPU DRAM ←→ Disk/Remote (Mooncake/Redis/NIXL)
- 是 vLLM v1 `kv_connector` 接口的最重要实现之一

### 1.4 vLLM v1 `kv_connector` 接口

- 文件：`vllm/distributed/kv_transfer/`
- 核心 API：`KVConnectorBase.send_kv_caches() / recv_kv_caches()`
- 后端：LMCache、Mooncake、NIXL

## 2. xPyD 弹性 PD（**2026 主流落地，必懂**）

### 2.1 是什么

xPyD = **x 个 prefill 实例 + y 个 decode 实例**（任意比例），动态调整。

经典 1P1D（一个 prefill 配一个 decode）已经不够灵活了：

- 长 prompt + 短 output 场景：prefill 卡死，decode 闲着 → 需要 3P1D
- 短 prompt + 长 output 场景：prefill 闲着 → 需要 1P3D

### 2.2 vLLM xPyD 实现

- ⭐⭐⭐ [vLLM PR #18242 · An native implementation of xPyD based on P2P NCCL](https://github.com/vllm-project/vllm/pull/18242) — 核心 PR
- ⭐⭐ [vLLM PR #12957 · Support XpYd disaggregated prefill with MooncakeStore](https://github.com/vllm-project/vllm/pull/12957) — Mooncake backend
- 架构：proxy/router 把请求路由到 1P1D 实例对，KV 走 P2P NCCL

### 2.3 SGLang xPyD 实现

- ⭐⭐⭐ [SGLang PD Disaggregation 官方文档（中文）](https://docs.sglang.com.cn/advanced_features/pd_disaggregation.html) — 含完整启动命令
- ⭐⭐ [SGLang Issue #9442 · Configuration of xPyD](https://github.com/sgl-project/sglang/issues/9442) — 配置示例
- ⭐⭐ [SGLang PR #18163 · improve kv offset calculation for MHA model with different tp size](https://github.com/sgl-project/sglang/pull/18163) — 实测 TTFT 从 899ms 降到 719ms

### 2.4 Transfer Backend 选型

| Backend | TP/DP 不同支持 | 适用场景 |
|---|---|---|
| Mooncake | ✅ 支持 | 生产部署、跨节点 RDMA |
| NIXL | ❌ 不支持 | 小规模、节点内 |
| P2P NCCL | ✅ 支持 | 默认 backend |

## 3. AF 分离（Attention/FFN Disaggregation）—— **2026 新方向**

### 3.1 为什么是 PD 分离之后的下一步

PD 分离解决了"prefill vs decode"矛盾，但**FFN 还是和 attention 在同一卡上**。

MoE 模型上，FFN（专家）算力 >>> attention：

- DeepSeek V3：671B 总参，FFN 占 80%+ 参数
- 但 attention 是 stateful（KV cache 大）、FFN 是 stateless

→ 把 attention 和 FFN 切到不同集群池

### 3.2 关键论文

- ⭐⭐⭐ **MegaScale-Infer (字节跳动, arXiv:2504.02263)** — [PDF](https://arxiv.org/pdf/2504.02263v2)
  - 为 MoE 模型提供 attention 和 FFN 分离的并行化策略
  - **Ping-pong 流水线**：在 attention 和 FFN 之间穿梭微批次
  - **M2N 通信库**：消除 GPU-CPU 数据拷贝和同步开销
  - 性能：相比 SOTA 提升 **1.90× 吞吐**
- ⭐⭐⭐ **Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving (arXiv:2601.21351)** — [PDF](https://arxiv.org/html/2601.21351v1)
  - 给出 AFD 资源配置比例的可解析框架
  - 理论最优 A/F 比例与模拟值差距 ≤10%
- ⭐⭐⭐ **百度 AFD 死区分析 (arXiv:2602.09721)** — [PDF](https://arxiv.org/pdf/2602.09721)
  - 重要发现："死区"现象：标准集群上增加 FFN 实例数无法改进 FLOPS 利用率
  - AFD 节点级离散扩展比 EP 连续 batch 调整产生更高的不平衡惩罚

### 3.3 vLLM 工程实现

- ⭐⭐⭐ [vLLM RFC #22799 · ATTN-FFN Disaggregation for MoE Models](https://github.com/vllm-project/vllm/issues/22799) — 2025.8 启动，目标支持 DeepSeek V3 + M2N 跨节点路由
- 这个 RFC 是 vLLM 2025-2026 最重要的架构演进之一，**值得长期跟踪**

### 3.4 适用场景

AFD 在以下条件下更具优势：
- ✅ Superpod 级硬件（高互连带宽）
- ✅ 专家粒度粗、稀疏度较低的模型
- ❌ 普通节点 / 低互连带宽场景下不如 EP

## 4. 大 EP（Large Expert Parallelism）—— **MoE 推理标配**

### 4.1 为什么需要"大 EP"

DeepSeek V3 / V4 / Qwen3.6 / Kimi K2 都是 MoE 模型，专家数大（256-384 个）：

- 小 EP（EP=8）：每卡 ~32-48 个专家，单卡承担多个专家计算量
- 大 EP（EP=64）：每卡 ~4-6 个专家，专家通信成主要开销

→ 需要专门的负载均衡和通信优化

### 4.2 EPLB（**DeepSeek 开源**）

- ⭐⭐⭐ [DeepSeek EPLB GitHub](https://github.com/deepseek-ai/EPLB) — Expert Parallelism Load Balancer
- ⭐⭐ [阿里云 · DeepSeek EPLB 冗余专家策略优化 MoE 模型负载均衡](https://developer.aliyun.com/article/1654261)
- ⭐⭐ [腾讯云 · DeepSeek 开源 EPLB 解读](https://deepseek.csdn.net/67fce31ea5baf817cf48e159.html)

**两种平衡策略**：
1. **分层负载均衡（prefill 阶段）**：先按节点分配专家组，节点内复制专家
2. **全局负载均衡（decode 阶段）**：全局复制专家，不限制专家组

### 4.3 实测性能（RTP-LLM 在阿里云灵骏环境）

- ⭐⭐⭐ [腾讯云 · 如何重现 DeepSeek 推理性能突破](https://cloud.tencent.com/developer/article/2523038) — RTP-LLM 实测
- Prefill：42.6K TPS（单节点 32 卡）
- Decode：14.7K TPS（单节点 144 卡）

### 4.4 关键技术点

- ⭐⭐⭐ [腾讯云 · EP 架构：DeepSeek 突破性实践背后](https://cloud.tencent.com.cn/developer/article/2504080) — EP 终极形态之争
- **MicroBatch + Overlapping**：前后分别遮盖 dispatch 和 combine 的通信时间
- **PD 分离 + EP 协同**：prefill 和 decode 用不同 TP/EP 配比
- **MTP 投机采样**：补偿 MicroBatch 带来的延迟

### 4.5 大 EP 的两难

| | 小 EP | 大 EP |
|---|---|---|
| 单卡显存 | 紧 | 松 |
| 通信量 | 小 | 大（all-to-all 占大头） |
| 负载均衡难度 | 低 | 高（必须配 EPLB） |
| 适合 | 16-64 卡 | 128-1024+ 卡 |

## 5. FlashInfer —— **推理 attention kernel 标杆库**

### 5.1 是什么

- ⭐⭐⭐ [flashinfer-ai/flashinfer GitHub](https://github.com/flashinfer-ai/flashinfer)
- 作者：Zihao Ye（叶子浩）— FlashAttention 论文 "From Online Softmax to FlashAttention" 的作者
- 专为大模型推理优化（区别于 FlashAttention 偏训练）

### 5.2 核心能力

- 高效稀疏/密集注意力 kernel（CUDA + Tensor Core）
- 向量稀疏注意力可达密集 kernel 带宽的 90%
- **JIT 编译支持自定义 attention 变体** —— 这意味着你可以快速试验新 attention（DSA、Gated DeltaNet 等）
- PageAttention 支持
- 高效 Top-P / Top-K 采样 kernel

### 5.3 vLLM / SGLang 怎么用

- vLLM v1 默认 attention backend 之一
- SGLang 内置 FlashInfer backend (`--mm-attention-backend fa3` 即用)
- DeepSeek V4 SGLang 部署用 FlashInfer TRTLLM-Gen MoE backend (MXFP8 × MXFP4)

### 5.4 中文资料

- ⭐⭐ [FlashInfer 源码级解读：大模型推理的"底层引擎"是怎么炼成的](https://www.yeyulingfeng.com/534685.html) — 含算子融合细节

## 6. 系统架构演进时间轴（**面试必背**）

```
2022.10  vLLM PagedAttention 论文
2023.10  vLLM v0 发布
2024.06  Mooncake 论文（KVCache-centric PD 分离）
2024.10  LMCache 多级 KV cache 工程实现
2025.01  vLLM v1 重构（统一 request lifecycle、原生多模态）
2025.04  MegaScale-Infer (字节，AF 分离开端)
2025.05  vLLM xPyD PR (#18242)
2025.08  vLLM ATTN-FFN Disaggregation RFC (#22799)
2025.09  DeepSeek V3.2 (DSA)
2025.11  vLLM-Omni 发布 (EPDG 全分离)
2026.04  DeepSeek V4 (CSA+HCA+mHC+Muon)
2026.04  Qwen3.6 (Gated DeltaNet 3:1 混合)
2026.Q1  vLLM-Omni Roadmap RFC #1192 (Omni Connector)
```

## 7. 面试可讲（**这一份让你比 99% 同行强**）

### Q1：PD 分离在什么场景下不如不分离？

A：
- 短 prompt + 短 output 场景：分离反而引入 KV 传输开销
- 单机小规模部署：跨卡 PD 比同卡 PD 慢
- 高 prefix cache 命中率场景：分离破坏了局部性

### Q2：xPyD 的最佳 P:D 比例怎么定？

A：
- 看 input/output 长度比：input/output ≈ 5:1 → P:D ≈ 1:3
- 看 SLO：TTFT-sensitive → 多 P；TBT-sensitive → 多 D
- 实际生产用动态调整（vLLM proxy/router 根据负载实时切换）

### Q3：AF 分离 vs EP 分离区别？

A：
- EP 分离：把不同**专家**放到不同卡，**横向**切 FFN
- AF 分离：把 attention 和 FFN 算子**纵向**分离到不同卡池
- 二者**正交**，可以同时使用（如 attention 节点 + 16 个 FFN 节点 EP=16）

### Q4：大 EP 为什么必须配 EPLB？

A：
- MoE 路由有"热门专家"现象（某些专家被频繁选中）
- 不平衡 → 部分卡 OOM、部分卡闲置
- EPLB 通过**冗余专家复制**把热门专家分散到多卡

### Q5：FlashInfer 和 FlashAttention 的核心区别？

A：
- FA 偏训练（前向 + 反向 + 梯度）
- FlashInfer 偏推理（PageAttention、JIT 自定义 kernel、Top-K 采样）
- FlashInfer 的稀疏 attention 在 V3.2 DSA 部署时是核心组件

## 8. 实操：跑通 xPyD（W16 stretch goal）

```bash
# Prefill instance (port 8001)
sglang serve <model> --port 8001 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device mlx5_0

# Decode instance (port 8002)
sglang serve <model> --port 8002 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device mlx5_1

# Proxy/Router (port 8000)
python -m sglang.launch_pd_router \
  --prefill-urls http://localhost:8001 \
  --decode-urls http://localhost:8002 \
  --port 8000
```

需要：
- 至少 2 张 GPU
- IB 网卡（推荐）或者 P2P NCCL（默认）

## 9. 论文阅读清单（按推荐顺序）

1. ⭐⭐⭐ vLLM Paper (SOSP'23, arXiv:2309.06180) — PagedAttention
2. ⭐⭐⭐ Mooncake (arXiv:2407.00079) — KVCache-centric PD
3. ⭐⭐⭐ MegaScale-Infer (arXiv:2504.02263) — AF 分离
4. ⭐⭐ Theoretically Optimal A/F Ratios (arXiv:2601.21351) — AFD 理论
5. ⭐⭐ AttentionStore (arXiv:2403.19708) — 多级 KV cache 早期工作
6. ⭐⭐ DistServe (OSDI'24) — PD 分离学术界标杆
7. ⭐⭐ DeepSeek-V3 / V4 技术报告 — 大 EP 工业界实践
