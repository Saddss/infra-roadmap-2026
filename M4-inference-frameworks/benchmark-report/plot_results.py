"""跑完 vllm/sglang 后用这个脚本画对比图"""

import json
import sys
from collections import defaultdict

import matplotlib.pyplot as plt


def main():
    # 把所有 results.jsonl 行读进来
    rows = []
    for path in sys.argv[1:] or ["results.jsonl"]:
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))

    # 按 framework × scenario 聚合
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["framework"], r["scenario"])].append(r)

    frameworks = sorted({r["framework"] for r in rows})
    scenarios = sorted({r["scenario"] for r in rows})

    # 画 throughput 对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    width = 0.35

    for ax, metric, ylabel in [
        (axes[0], "ttft", "TTFT p50 (ms)"),
        (axes[1], "throughput", "Throughput (tok/s, computed)"),
    ]:
        for i, fw in enumerate(frameworks):
            vals = []
            for sc in scenarios:
                items = grouped.get((fw, sc), [])
                if not items:
                    vals.append(0); continue
                if metric == "ttft":
                    s = sorted([it["ttft"] for it in items])
                    vals.append(s[len(s) // 2] * 1000)
                else:
                    total_tok = sum(it["output_tokens"] for it in items)
                    total_time = sum(it["end_to_end"] for it in items)
                    vals.append(total_tok / total_time if total_time else 0)
            x = [j + i * width for j in range(len(scenarios))]
            ax.bar(x, vals, width, label=fw)
        ax.set_xticks([j + width / 2 for j in range(len(scenarios))])
        ax.set_xticklabels(scenarios, rotation=15)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("benchmark_comparison.png", dpi=120)
    print("图已保存到 benchmark_comparison.png")


if __name__ == "__main__":
    main()
