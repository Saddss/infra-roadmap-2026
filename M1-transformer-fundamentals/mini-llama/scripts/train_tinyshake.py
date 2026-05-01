"""
W4 任务：在 tinyshakespeare 上跑通训练。

这是一个最小训练 loop 模板，你的 MiniLlama 实现完后跑这个。
跑通 = M1 任务完成的一个标志。

用法:
    cd mini-llama
    python scripts/train_tinyshake.py --attn gqa --steps 500
"""

import argparse
import os
import sys
import urllib.request

import torch
import torch.nn.functional as F

from mini_llama.model import MiniLlama, ModelConfig


SHAKE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "data/tinyshake.txt"


def download_data():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print(f"下载数据集到 {DATA_PATH} ...")
        urllib.request.urlretrieve(SHAKE_URL, DATA_PATH)
    return open(DATA_PATH).read()


def build_char_tokenizer(text):
    chars = sorted(list(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join(itos[i] for i in l)
    return encode, decode, len(chars)


def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attn", choices=["mha", "gqa", "mla"], default="gqa")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq", type=int, default=128)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = download_data()
    encode, decode, vocab_size = build_char_tokenizer(text)
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    config = ModelConfig(
        vocab_size=vocab_size,
        dim=192, n_layers=4, n_heads=6, n_kv_heads=2,
        max_seq_len=args.seq, attention_type=args.attn,
    )
    model = MiniLlama(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params / 1e6:.2f}M params, attn={args.attn}, device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for step in range(args.steps):
        x, y = get_batch(train_data, args.batch, args.seq, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 50 == 0:
            with torch.no_grad():
                vx, vy = get_batch(val_data, args.batch, args.seq, device)
                vl = F.cross_entropy(model(vx).view(-1, vocab_size), vy.view(-1))
            print(f"step {step:>4}  train {loss.item():.3f}  val {vl.item():.3f}")

    # 生成几句话证明能用
    print("\n--- Sample ---")
    prompt = torch.tensor([[encode("ROMEO:")[0]]], device=device)
    out = model.generate(prompt, max_new_tokens=200, temperature=0.8)
    print(decode(out[0].tolist()))


if __name__ == "__main__":
    main()
