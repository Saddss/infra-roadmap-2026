# vLLM / SGLang / TRT-LLM 推理框架横向 Benchmark 报告

> 作者：你
> 日期：YYYY-MM-DD
> 硬件：1× / 4× <GPU 型号>
> 模型：Qwen2.5-7B-Instruct（建议用这个，体量适中、社区认可度高）

## 总览（结论先行）

| 框架 | 主打优势 | 适合场景 | 不适合场景 |
|---|---|---|---|
| vLLM | 通用、生态广 |  |  |
| SGLang | RadixAttention、Agent 场景 |  |  |
| TRT-LLM | NVIDIA 全栈极致性能 |  |  |

我的实测结论一句话：

```
（写一句你的核心 finding，比如 "对于多轮对话场景，SGLang 比 vLLM 平均快 X%"）
```

## 1. 实验设计

### 1.1 硬件
### 1.2 模型
### 1.3 测试集（prompts）
- 短 prompt（<256 tokens）
- 长 prompt（>4K tokens）
- 多轮对话（共享 system prompt）
- Agent 风格（JSON tool call）

### 1.4 指标
- Throughput (tokens/sec)
- TTFT (Time to First Token)
- TPOT (Time Per Output Token)
- p50 / p99 延迟

## 2. Throughput 对比

（贴图）

## 3. 延迟对比

（贴图）

## 4. 显存占用对比

（贴图）

## 5. 量化方案对比（vLLM 内）

| 方案 | throughput | accuracy (MMLU) | 显存 |
|---|---|---|---|
| fp16 |  |  |  |
| awq |  |  |  |
| gptq |  |  |  |
| fp8 |  |  |  |

## 6. Speculative Decoding 收益

| 场景 | no spec | with spec | acceptance rate |
|---|---|---|---|
| 短输出 |  |  |  |
| 长输出 |  |  |  |
| 多轮对话 |  |  |  |

## 7. Prefix Caching 收益

固定 system prompt + 100 个不同 user query：

| 框架 | first request | last request | 加速比 |
|---|---|---|---|
| vLLM (no prefix cache) |  |  |  |
| vLLM (with prefix cache) |  |  |  |
| SGLang |  |  |  |

## 8. 我的选型建议

### 你的业务是什么类型 → 选什么框架？

- 通用 OpenAI API 服务 → vLLM
- 大量 Agent / 多轮对话 → SGLang
- 极致延迟 / NVIDIA 全栈 → TRT-LLM
- 边缘部署 / Mac → llama.cpp / Ollama

## 9. 简历可写的一句话

> 完成 vLLM / SGLang / TRT-LLM 三大主流推理框架的端到端 benchmark 对比，覆盖 throughput / TTFT / 量化 / spec decoding / prefix caching 5 大维度，发布详细技术报告。在 X 场景下证明 SGLang 比 vLLM 提升 Y%。
