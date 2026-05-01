#!/usr/bin/env bash
# W17 任务：4 卡跑通 Megatron GPT-2 训练
#
# 前置:
#   1. clone NVIDIA/Megatron-LM
#   2. 准备一个小数据集（比如 wikipedia 子集）
#   3. 4 张 NVIDIA GPU（A10/A100/4090，内存 ≥ 24GB）
#
# 实验目标：分别跑 TP=1/2/4，对比 throughput

set -euo pipefail

MEGATRON_REPO=${MEGATRON_REPO:-$HOME/Megatron-LM}
DATA_PATH=${DATA_PATH:-$HOME/data/wikipedia/wiki_text_document}
TOKENIZER=${TOKENIZER:-$HOME/data/gpt2-vocab.json}
MERGES=${MERGES:-$HOME/data/gpt2-merges.txt}

cd "$MEGATRON_REPO"

# === 实验 1: TP=4, PP=1 ===
torchrun --nproc-per-node=4 --master-port=6000 pretrain_gpt.py \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 1 \
    --sequence-parallel \
    --num-layers 12 --hidden-size 1024 --num-attention-heads 16 \
    --seq-length 1024 --max-position-embeddings 1024 \
    --micro-batch-size 4 --global-batch-size 32 \
    --train-iters 200 --lr 1e-4 \
    --data-path "$DATA_PATH" --vocab-file "$TOKENIZER" --merge-file "$MERGES" \
    --tokenizer-type GPT2BPETokenizer \
    --use-flash-attn \
    --bf16 \
    --log-interval 10 \
    2>&1 | tee tp4_pp1.log

# 同样模式跑 TP=2/PP=2、TP=1/PP=4 三组实验
# 然后对比 samples/sec

echo "=== 完成 4 卡 Megatron 实验 ==="
echo "查看 log: tp4_pp1.log"
echo ""
echo "下一步：用 grep 'samples/sec' tp*.log 提取吞吐数据，画对比图"
