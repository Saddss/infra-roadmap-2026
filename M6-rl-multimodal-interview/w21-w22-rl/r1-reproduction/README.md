# R1 类复现（GRPO + Qwen2.5-1.5B）

## 文件说明

- `run_grpo_qwen_countdown.sh` — 启动脚本（OpenRLHF 版）
- `reward_function.py` — Countdown 任务奖励函数（已通过测试）
- `notes.md` — 你跑通后填的踩坑记录

## 跑通流程

### 1. 准备环境

```bash
# 创建独立环境
conda create -n openrlhf python=3.10 -y
conda activate openrlhf

# 安装
pip install openrlhf
pip install vllm  # 用作 rollout engine

# 验证
python -c "import openrlhf; print('ok')"
```

### 2. 准备数据

```bash
# Countdown-Tasks-3to4 数据集（HuggingFace）
huggingface-cli download Jiayi-Pan/Countdown-Tasks-3to4 --repo-type=dataset
```

### 3. 启动 Ray + 训练

```bash
# 假设有 3 张 A800 80GB
ray start --head --num-gpus=3

bash run_grpo_qwen_countdown.sh
```

### 4. 测试训完的模型

```bash
# 用 vLLM 起 server
vllm serve ./grpo-qwen2.5-1.5b-countdown --port 8001

# 测试
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{
      "role": "user",
      "content": "Use [19, 36, 55, 7] with +-*/() to get 65. Show your reasoning in <think>...</think> and answer in <answer>...</answer>."
    }]
  }'
```

期望看到：模型能输出 `<think>...let me try ...</think>\n<answer>19 + 36 + 10</answer>` 这样的格式。

## 备选方案：用 TinyZero（更省钱）

如果 OpenRLHF 跑不顺，TinyZero 用 veRL：

```bash
git clone https://github.com/Jiayi-Pan/TinyZero
cd TinyZero
# 按 README 跑
```

只需 4 张 A800 + 8 小时 ≈ 224 元。
