#!/usr/bin/env bash
# W22 任务：用 GRPO 训练 Qwen2.5-1.5B-Instruct 解 Countdown 数学游戏
#
# 这是 R1 风格的最小复现。3×A800 80GB 跑约 20 小时，约 420 元云费。
#
# 前置:
#   pip install vllm openrlhf
#   或: pip install transformers trl deepspeed
# 数据集:
#   Jiayi-Pan/Countdown-Tasks-3to4 (HF)

set -euo pipefail

MODEL=Qwen/Qwen2.5-1.5B-Instruct
DATASET=Jiayi-Pan/Countdown-Tasks-3to4
OUT_DIR=./grpo-qwen2.5-1.5b-countdown
NUM_GPUS=3

# === 选项 A: 用 OpenRLHF（推荐） ===
ray start --head --num-gpus=$NUM_GPUS

python -m openrlhf.cli.train_ppo_ray \
  --pretrain $MODEL \
  --advantage_estimator group_norm \
  --num_episodes 1 \
  --train_batch_size 32 \
  --rollout_batch_size 64 \
  --num_prompts_per_episode 16 \
  --num_responses_per_prompt 8 \
  --max_samples 20000 \
  --prompt_max_len 256 \
  --max_new_tokens 4096 \
  --actor_learning_rate 5e-7 \
  --kl_coef 0.001 \
  --vllm_num_engines 1 \
  --vllm_tensor_parallel_size 1 \
  --colocate_actor_ref \
  --remote_rm_url ./reward_function.py \
  --dataset $DATASET \
  --save_path $OUT_DIR \
  --logging_steps 1 \
  --eval_steps 50 \
  --save_steps 100 \
  --use_wandb $WANDB_API_KEY

echo "训练完成。模型保存在 $OUT_DIR"
echo "下一步：用 vllm serve 跑跑看效果"
