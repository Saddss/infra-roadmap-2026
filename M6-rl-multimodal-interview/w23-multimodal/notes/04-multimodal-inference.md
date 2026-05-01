# 多模态推理优化

## vLLM v1 的多模态原生支持

v1 的关键改进：

```
v0: vision encoder 是后期补丁，单独排队
v1: vision encoder 和 LLM decoder 共享调度器
    → 同一个 batch 里既有纯文本请求又有图像请求
    → 视觉 token 也能 prefix caching
```

## SGLang 的多模态优势

特别强：

1. **动态分辨率原生支持**：不同尺寸图像 → 不同 patch 数
2. **跨图像 attention**：多图场景下，图像间 KV cache 也享 RadixAttention 优化
3. **视频帧序列**：少数支持视频帧序列推理的框架之一

## Prefix Caching for 多模态

```
场景：同一张图 + 不同 user query
v0: 视觉 token 全部重新算
v1 / SGLang: 视觉 token 缓存复用，只算 user query 部分
```

→ 多模态 Agent 场景下 prefix caching 红利非常大。

## 跑通 Qwen2.5-VL（W23 实操）

```bash
pip install vllm>=0.5.0

# 启动
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --port 8000 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85

# 给图测试
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "messages": [{
      "role": "user",
      "content": [
        {"type":"text","text":"What is in this image?"},
        {"type":"image_url","image_url":{"url":"https://..."}}
      ]
    }]
  }'
```

## 我的发现

```
（实测：图像越大、Q 越短，prefix caching 收益越大，因为视觉 token 占比大）
```

## 面试可讲

- Q1：多模态推理的 KV cache 怎么管？
- Q2：动态分辨率会带来什么调度问题？
- Q3：多图场景下 SGLang 为什么比 vLLM 快？
- Q4：DiT 推理能复用 LLM 推理框架吗？
