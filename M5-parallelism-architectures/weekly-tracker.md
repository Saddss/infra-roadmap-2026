# M5 周记 · 并行 + 2026 新架构

## W17 周记 · DP / TP / PP 主线

- [ ] Megatron-LM clone + 4 卡跑通 GPT-2 训练
- [ ] 分别跑 (TP=1, PP=4) / (TP=2, PP=2) / (TP=4, PP=1) 对比 throughput
- [ ] notes/01-data-parallel.md / 02-tensor-parallel.md / 03-pipeline-parallel.md 写完

**实验数据**：

```
GPT-2-1.3B 4 卡:
  TP=1, PP=4: __ samples/s
  TP=2, PP=2: __ samples/s
  TP=4, PP=1: __ samples/s
```

**焦虑分**：1-10

---

## W18 周记 · CP / EP / SP / Muon

- [ ] 4 份笔记完成（04-context / 05-expert / 06-sequence / 07-muon）
- [ ] FSDP vs DeepSpeed vs Megatron 对比表写到博客里

**焦虑分**：1-10

---

## W19 周记 · DeepSeek 系（V3 → V3.2 → V4）

- [ ] V3 报告 Section 2.1 + V3.2 + V4 三代连读
- [ ] `deepseek-v3-v32-v4/notes.md` 写完
- [ ] DSA Lightning Indexer 50-80 行 PyTorch 复现 → `dsa_lightning_indexer.py`
- [ ] （可选）跑通 V3.2-Exp 或 V4-Flash 推理

**核心洞察**：

```
（V3 → V4 这条路线的内在逻辑是什么？）
```

**焦虑分**：1-10

---

## W20 周记 · Qwen 系 + Kimi/MiniMax + 综述博客

- [ ] Qwen3 → Qwen3-Next → Qwen3.6 三代连读
- [ ] `qwen3-qwen36/notes.md` 写完
- [ ] Gated DeltaNet 50-80 行 PyTorch 复现 → `gated_deltanet.py`
- [ ] Kimi K2 Thinking + Kimi Linear + MiniMax M2 笔记
- [ ] **综述博客发布**《2026 大模型架构两大流派 · V4 vs Qwen3.6》

**博客链接**：

```
（粘 URL）
```

**博客评论数**：__（≥ 5 才能进 M6）

**M5 总结**（500 字写给一个月后的自己）：

```
（如果让你给一个想入推理工程师的同事 30 分钟讲清楚 2026 架构演进，你怎么讲？）
```

**焦虑分**：1-10
