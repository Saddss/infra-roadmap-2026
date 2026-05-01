# M4（W13-W16）：推理框架横向 + 工程优化武器库

## 目标

除了 vLLM，还能讲清 SGLang/TRT-LLM 的差异、各自适合什么场景，并对量化、speculative decoding、disaggregated prefill 这些"加分项"有动手经验。

**这一段是简历最值钱的部分** —— 把 benchmark 数据写进简历那一栏。

## 必交产出物

- [ ] **完整 benchmark 报告**（在 `benchmark-report/` 输出 PDF / 长文）
- [ ] **1-2 个 Tier 2/3 PR**（看 `pr-tracker/`）
- [ ] 4 个专题各一篇短笔记

## 每周拆解（一周一个专题）

### W13：SGLang RadixAttention

- [ ] clone `sgl-project/sglang`，跑通官方 demo
- [ ] 读 `sglang/srt/managers/cache_controller.py` 等核心文件
- [ ] 跑 prefix sharing benchmark：固定 system prompt + 1000 个不同 user query
- [ ] 写 `w13-sglang-radixattention/notes.md`：RadixAttention vs vLLM PagedAttention 的本质区别

### W14：Speculative Decoding

- [ ] 读综述：知乎搜 "投机采样综述"
- [ ] 理解 Medusa / EAGLE / MTP 三种范式
- [ ] 在 vLLM 里启用 spec decode：`--speculative-model <draft_model>`
- [ ] 测 throughput + acceptance rate，写 `w14-speculative-decoding/notes.md`

### W15：量化（FP8 / AWQ / GPTQ / INT4 KV Cache）

- [ ] 跑通 `vllm serve --quantization awq <model>`
- [ ] 对比同一模型 fp16 vs awq vs gptq 的 throughput / accuracy（用 LM-eval-harness）
- [ ] 试 INT4 KV Cache（vLLM 新支持的 `--kv-cache-dtype int4`）
- [ ] 写 `w15-quantization/notes.md`

### W16：Disaggregated Prefill / Multi-LoRA + 提交 Tier 2/3 PR

- [ ] 读 vLLM v1 的 `kv_connector` 接口设计
- [ ] 试 `--enable-prefix-caching --enable-chunked-prefill`
- [ ] 试 Multi-LoRA：同时加载多个 LoRA adapter，按请求路由
- [ ] **提交 1-2 个 Tier 2/3 PR**（小 bug fix / 新 sampler / benchmark 改进）
- [ ] 完成 benchmark 总报告

## 阶段切换检查表（进 M5 前）

- [ ] benchmark 报告完整发布（建议同时发知乎 + GitHub）
- [ ] 简历更新："基于 vLLM 实现 XXX 优化，benchmark 显示 throughput 提升 X% / latency 降低 Y%"
- [ ] 至少 1 个 Tier 2/3 PR 已提交

## 资料

→ 见 [`resources.md`](./resources.md)
