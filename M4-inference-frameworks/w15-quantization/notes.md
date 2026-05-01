# W15: 量化笔记

## 量化方法对比

| 方法 | 压缩谁 | 精度损失 | 推理加速 | 训练改动 | 部署难度 |
|---|---|---|---|---|---|
| FP16 → BF16 | 权重+激活 | 几乎无 | 1x | 无 | 易 |
| FP8 (E4M3/E5M2) | 权重+激活+KV | 极小 | 1.5-2x | 训练时介入 | 中 |
| AWQ | 权重 (W4A16) | 小 | 2-3x | 量化校准 | 易 |
| GPTQ | 权重 (W4A16) | 小 | 2-3x | 量化校准 | 易 |
| INT4 KV Cache | 仅 KV | 中 | 1.3x | 无 | 易 |
| SmoothQuant | 权重+激活 (W8A8) | 中 | 2x | 校准 | 中 |

## 实验：Qwen2.5-7B 量化对比

```bash
# fp16 baseline
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# AWQ
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq --port 8001

# fp8（H100）
vllm serve Qwen/Qwen2.5-7B-Instruct --quantization fp8 --port 8002

# INT4 KV Cache
vllm serve Qwen/Qwen2.5-7B-Instruct --kv-cache-dtype fp8_e5m2 --port 8003
```

精度评测：

```bash
pip install lm-eval
lm_eval --model openai-completions \
  --model_args base_url=http://localhost:8000/v1,model=Qwen2.5-7B \
  --tasks mmlu,gsm8k --num_fewshot 0
```

## 实验数据

| 方案 | MMLU | GSM8K | throughput | 显存 |
|---|---|---|---|---|
| fp16 |  |  |  |  |
| awq |  |  |  |  |
| gptq |  |  |  |  |
| fp8 |  |  |  |  |
| int4 KV |  |  |  |  |

## 我的发现

```
（哪个方法在你的硬件上 ROI 最高？为什么 H100 推荐 fp8？）
```
