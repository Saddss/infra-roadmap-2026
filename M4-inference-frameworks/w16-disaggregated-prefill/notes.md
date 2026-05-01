# W16: Disaggregated Prefill / Multi-LoRA 笔记

## Disaggregated Prefill 是什么

```
（一段话）
```

**Why**：

- prefill 是 compute-bound，decode 是 memory-bound
- 把它们放在同一卡上 → 互相干扰、利用率低
- 分离部署 → prefill 用满计算单元，decode 用满 memory bandwidth

## vLLM 的 KV Connector 接口

文件：`vllm/distributed/kv_transfer/`

核心 API：
- `KVConnectorBase.send_kv_caches(...)`  prefill 端导出 KV
- `KVConnectorBase.recv_kv_caches(...)`  decode 端拉取 KV

可选 backend：
- LMCache
- Mooncake
- NIXL

## Multi-LoRA 实验

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --enable-lora \
  --lora-modules \
    sql-lora=path/to/sql-adapter \
    math-lora=path/to/math-adapter \
    code-lora=path/to/code-adapter
```

请求时通过 `model` 字段指定要走哪个 LoRA：

```python
client.chat.completions.create(model="sql-lora", ...)
```

## 实验数据

```
3 个 adapter 同时加载，每秒交替请求：
  no LoRA          : __ tok/s
  with LoRA (single): __ tok/s
  with LoRA (multi) : __ tok/s
  3 个独立服务      : __ tok/s（这个最优但成本高）
```

## 我的结论

```
```
