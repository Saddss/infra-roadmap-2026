# V4 / Qwen3.6 跑通笔记

W20 任务：把这两个 2026 旗舰模型至少跑通一次推理。

## 选项 A：跑 Qwen3.6-35B-A3B（24GB GPU 即可）

最经济选择。一张 4090 24GB + 4-bit 量化就能跑。

```bash
# 准备
pip install vllm>=0.19.0 transformers>=4.45.0

# 启动
vllm serve Qwen/Qwen3.6-35B-A3B \
  --port 8000 \
  --max-model-len 32768 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --quantization awq

# 测试
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.6-35B-A3B",
    "messages": [{"role":"user","content":"用 Python 写一个快排"}]
  }'
```

## 选项 B：跑 DeepSeek V4-Flash（13B 激活，需 H100）

显存要求高，云租 1×H100 80G 几小时即可。

```bash
# 准备
pip install vllm>=0.19.5  # vLLM 加 V4 支持的版本

# 启动
vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --port 8000 \
  --max-model-len 65536 \
  --tensor-parallel-size 1
```

## 我的跑通记录（W20 填）

### 时间 / 硬件

```
日期: 2026-XX-XX
GPU:  RTX 4090 24GB / H100 80GB
模型: Qwen3.6 / V4-Flash
```

### 踩坑

| 坑 | 现象 | 怎么解决 |
|---|---|---|
|  |  |  |

### 性能数据

```
prompt: 128 tokens
output: 256 tokens

throughput: __ tok/s
TTFT: __ ms
显存峰值: __ GB
```

### 一段质量测试

```
（让模型回答几个测试 prompt，把回答粘进来证明你真跑通了）

Q1: 帮我用 Python 实现 GQA attention
A1: ...

Q2: 你和 Qwen3 的区别是什么？
A2: ...
```

### 我的发现 / 面试可讲

```
（部署时遇到的真实问题，比如 vLLM 哪个版本 bug、量化掉点多少等）
```
