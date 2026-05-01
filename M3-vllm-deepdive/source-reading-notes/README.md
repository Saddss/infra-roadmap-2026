# vLLM 源码笔记

每读一个核心文件，写一份笔记到这个目录。模板：

## 笔记模板（复制下面这段开始）

```markdown
# `<文件路径>` 笔记

> 阅读日期：YYYY-MM-DD
> commit hash：xxxxxxx

## 这个文件解决什么问题
（一句话）

## 关键 class / 函数
- `ClassA.method_x()` —— 一句话职责
- ...

## 数据流（画时序图）
（mermaid sequenceDiagram）

## 我画的工作图
（mermaid flowchart 或 erDiagram）

## 关键代码片段（带注释）
\`\`\`python
（粘 5-20 行最关键代码，逐行加你的理解注释）
\`\`\`

## 我的疑问
- Q1:
- Q2:

## 后续要读什么
（这个文件依赖了哪些其他文件？读完应该接着读什么？）
```

## 推荐顺序

1. `01-engine-core.md` — 控制中枢
2. `02-scheduler.md` — 调度策略
3. `03-kv-cache-manager.md` — Block Table
4. `04-gpu-model-runner.md` — forward 执行
5. `05-paged-attention-kernel.md` — CUDA kernel
