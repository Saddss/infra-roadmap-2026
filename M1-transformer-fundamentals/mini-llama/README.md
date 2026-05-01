# mini-llama

> A minimal-from-scratch LLaMA implementation that supports MHA / MQA / GQA / MLA attention variants.
> Built as a learning project to truly understand the evolution of attention mechanisms.

## Structure

```
mini_llama/
  attention/
    single_head.py      # W1: numpy single-head attention
    mha.py              # W2: multi-head attention
    mqa.py              # W2: multi-query attention
    gqa.py              # W2: grouped-query attention
    mla.py              # W3: multi-head latent attention (DeepSeek)
  positional/
    rope.py             # W3: rotary position embedding
  norm/
    rms_norm.py         # W4: RMSNorm
  model.py              # W4: assembled mini-LLaMA model
tests/
  test_*.py             # unit tests, must pass
scripts/
  compare_kv_cache.py   # W2: KV cache memory comparison
  train_tinyshake.py    # W4: training loop on tinyshakespeare
notes/
  *.md                  # your derivation notes
```

## Setup

```bash
cd ~/sss/infra-roadmap-2026/M1-transformer-fundamentals/mini-llama
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run tests

```bash
pytest tests/ -v
```

## Roadmap (4 weeks)

- [ ] W1: single-head attention (numpy) + tests pass
- [ ] W2: MHA / MQA / GQA + KV cache comparison script
- [ ] W3: MLA + RoPE
- [ ] W4: assemble + train + blog post

## What I learned

(fill in W4)
