"""
W11 实验：vLLM throughput-latency 曲线

跑这个脚本会把 vLLM 在不同 max_num_batched_tokens 配置下的吞吐和延迟画出来。
图直接放在博客《vLLM 一个请求的一生》末尾。

前置:
    pip install vllm
    需要至少 1 张 24GB+ GPU

用法:
    python bench_vllm_throughput_latency.py --model Qwen/Qwen2.5-1.5B-Instruct
"""

import argparse
import time

import matplotlib.pyplot as plt


def bench_one_config(model: str, max_num_batched_tokens: int, n_requests: int = 64):
    """跑一组配置，返回 (throughput tok/s, p50_latency, p99_latency)"""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("请先 pip install vllm")
        return None

    llm = LLM(
        model=model,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=0.85,
        enforce_eager=True,  # 跳过 CUDA Graph 编译，加快启动
    )

    # 简单 prompt
    prompts = ["Tell me about Beijing in 100 words."] * n_requests
    sp = SamplingParams(max_tokens=128, temperature=0.7)

    start = time.perf_counter()
    outs = llm.generate(prompts, sp)
    dur = time.perf_counter() - start

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    throughput = total_tokens / dur

    latencies = sorted([o.metrics.finished_time - o.metrics.arrival_time for o in outs])
    p50 = latencies[len(latencies) // 2]
    p99 = latencies[int(len(latencies) * 0.99)]

    return throughput, p50, p99


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = p.parse_args()

    configs = [512, 1024, 2048, 4096, 8192]

    throughputs, p50s, p99s = [], [], []
    for c in configs:
        print(f"\n=== max_num_batched_tokens = {c} ===")
        result = bench_one_config(args.model, c)
        if result is None:
            return
        tps, p50, p99 = result
        print(f"  throughput {tps:.1f} tok/s | p50 {p50*1000:.0f} ms | p99 {p99*1000:.0f} ms")
        throughputs.append(tps); p50s.append(p50 * 1000); p99s.append(p99 * 1000)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.plot(configs, throughputs, "o-", color="tab:blue", label="throughput")
    ax2.plot(configs, p50s, "s--", color="tab:orange", label="p50 latency")
    ax2.plot(configs, p99s, "^--", color="tab:red", label="p99 latency")

    ax1.set_xscale("log"); ax1.set_xlabel("max_num_batched_tokens")
    ax1.set_ylabel("throughput (tok/s)", color="tab:blue")
    ax2.set_ylabel("latency (ms)", color="tab:red")
    plt.title(f"vLLM throughput vs latency  (model={args.model})")
    fig.legend(loc="upper center", ncol=3)
    fig.tight_layout()
    plt.savefig("vllm_throughput_latency.png", dpi=120)
    print("\n图已保存到 vllm_throughput_latency.png")


if __name__ == "__main__":
    main()
