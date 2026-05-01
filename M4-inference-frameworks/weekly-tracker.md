# M4 周记 · 推理框架横向

## W13 周记 · SGLang RadixAttention

**完成度**：

- [ ] SGLang clone + 跑通 demo
- [ ] 读 cache_controller.py / radix_cache.py
- [ ] prefix sharing benchmark
- [ ] notes.md 写完

**核心洞察**：

```
（PagedAttention 是 block 级 hash，RadixAttention 是 token 级前缀树。各自适合什么？）
```

**焦虑分**：1-10

---

## W14 周记 · Speculative Decoding

**完成度**：

- [ ] 投机采样综述读完
- [ ] Medusa / EAGLE / MTP 三种范式区分清楚
- [ ] 在 vLLM 里跑通 spec decode
- [ ] 测 throughput + acceptance rate

**实验数据**：

```
target=Qwen2.5-7B, draft=Qwen2.5-0.5B, prompt=128, output=512
  no spec   : ___ tok/s
  with spec : ___ tok/s
  acceptance rate : ___ %
```

**焦虑分**：1-10

---

## W15 周记 · 量化

**完成度**：

- [ ] AWQ / GPTQ / fp8 / int4 跑通对比
- [ ] LM-eval-harness 测精度
- [ ] notes.md 写完

**实验数据**：

```
Qwen2.5-7B on MMLU:
  fp16  : __ % accuracy, __ tok/s, __ GB
  awq   : __ % accuracy, __ tok/s, __ GB
  gptq  : __ % accuracy, __ tok/s, __ GB
  fp8   : __ % accuracy, __ tok/s, __ GB
  int4 KV: __ % accuracy, __ tok/s, __ GB
```

**焦虑分**：1-10

---

## W16 周记 · Disaggregated + PR + 总结

**完成度**：

- [ ] kv_connector 接口设计读完
- [ ] Multi-LoRA 跑通（同时加载 3 个 adapter）
- [ ] **Tier 2/3 PR 提交**
- [ ] benchmark 总报告完成

**PR 链接**：

```
PR #__:
```

**报告链接**：

```
（发到知乎 / 个人博客）
```

**M4 总结**（300 字写给一个月后的自己）：

```
```

**焦虑分**：1-10
