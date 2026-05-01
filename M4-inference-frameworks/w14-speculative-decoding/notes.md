# W14: Speculative Decoding 笔记

## 三种范式区分

| | Medusa | EAGLE | MTP |
|---|---|---|---|
| 出处 | Medusa 论文 | EAGLE 论文 | DeepSeek-V3 |
| 谁出 draft tokens |  |  |  |
| Verify 阶段 |  |  |  |
| 训练成本 |  |  |  |
| Acceptance rate |  |  |  |
| 适合场景 |  |  |  |

## 在 vLLM 里跑通 spec decode

```bash
# 用 7B 主模型 + 0.5B draft 模型
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --speculative-model Qwen/Qwen2.5-0.5B \
  --num-speculative-tokens 4 \
  --port 8000
```

## 实验数据

**配置**：Qwen2.5-7B + draft Qwen2.5-0.5B，prompt 128 tok，max_output 512 tok

| 配置 | throughput | acceptance rate | TTFT |
|---|---|---|---|
| no spec | __ tok/s | - | __ ms |
| spec, k=2 | __ | __ % | __ ms |
| spec, k=4 | __ | __ % | __ ms |
| spec, k=8 | __ | __ % | __ ms |

## 我的发现

```
（k 多大时收益开始下降？为什么？）
```
