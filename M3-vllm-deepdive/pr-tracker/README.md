# PR 跟踪

每提一个 PR，新建一个文件 `prNNN-<title>.md`。

## 模板

```markdown
# PR #NNN: <title>

- **仓库**：vllm-project/vllm
- **链接**：https://github.com/vllm-project/vllm/pull/NNN
- **类型**：Tier 0 / 1 / 2 / 3 / 4
- **目标 issue**：#MMM
- **状态**：open / merged / closed / changes-requested
- **提交日期**：YYYY-MM-DD
- **首次响应日期**：YYYY-MM-DD
- **merge 日期**：YYYY-MM-DD（如已 merge）

## 描述

（一段话说明改了什么）

## Review 历程

| 日期 | reviewer | 反馈类型 | 我做了什么 |
|---|---|---|---|
|  |  |  |  |

## 学到的东西

（被 review 后才理解的东西、维护者风格、设计偏好）
```

## 6 个月目标 PR 数量

| 阶段 | 目标 | 已提交 | 已合并 |
|---|---|---|---|
| M3 | 2-3 个 Tier 0/1 |  |  |
| M4 | 1-2 个 Tier 2/3 |  |  |
| M5 | 0-1 个 Tier 3/4 |  |  |
| **总计** | **3-6 个** |  |  |

## 找 issue 的固定 SOP（每周日 30 min）

1. 进 [vllm-project/vllm Issues · good first issue](https://github.com/vllm-project/vllm/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
2. 排除 label `stale`
3. 按 Newest 排序，看前 20 个
4. 筛选条件：
   - 描述清晰，有 expected/actual behavior
   - 已经有 reproducer
   - 评论数 < 5（多了说明已经讨论烂了）
   - 没有 "I'll take this" 的评论
5. 评论：`I'd like to take this. Will submit a PR within 5 days.`
6. 5 天内必须出 PR
7. 同时去看最近 7 天 merge 的 PR，学维护者的 review 风格

## 备选目标仓库（vLLM 太卷时）

按"接受率 × 简历加分"排序：

| 仓库 | 路径 | 推荐方向 |
|---|---|---|
| **SGLang** | `sgl-project/sglang` | 同 vLLM，更包容 |
| **OpenRLHF** | `OpenRLHF/OpenRLHF` | 算法变体、文档 |
| **Qwen** | `QwenLM/Qwen3` | demo / cookbook |
| **DeepSeek** | `deepseek-ai/*` | inference 优化 |
| **FlashInfer** | `flashinfer-ai/flashinfer` | kernel 微优化 |
| **llama.cpp** | `ggerganov/llama.cpp` | 新模型 GGUF 支持 |
