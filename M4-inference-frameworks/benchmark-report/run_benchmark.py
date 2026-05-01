"""
M4 benchmark 主脚本：统一跑 vLLM / SGLang，输出 throughput / TTFT / TPOT

设计原则：用 OpenAI 兼容 API 跑（vLLM 和 SGLang 都支持），
框架之间保持公平。

用法:
    # 先在另一个终端启动 vLLM serve:
    #   vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
    # 然后跑 benchmark:
    python run_benchmark.py --base-url http://localhost:8000/v1 --label vllm

    # 同样方式跑 SGLang，--label sglang
"""

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, asdict
from typing import List

import aiohttp


@dataclass
class RequestStats:
    ttft: float
    end_to_end: float
    output_tokens: int
    framework: str
    scenario: str


PROMPT_SHORT = "Tell me about Beijing."
PROMPT_LONG = "Summarize the following text:\n" + ("Beijing is the capital of China. " * 200)
PROMPT_SYS = "You are a helpful assistant. Always answer in detail."


async def one_request(session, base_url, model, prompt, system_prompt, scenario, framework):
    """发一个请求，记录 TTFT 和 end-to-end 时间"""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "max_tokens": 256,
        "temperature": 0.7,
    }

    start = time.perf_counter()
    ttft = None
    output_tokens = 0
    async with session.post(f"{base_url}/chat/completions", json=payload) as resp:
        async for line in resp.content:
            line = line.decode("utf-8").strip()
            if not line.startswith("data:"): continue
            chunk = line[5:].strip()
            if chunk == "[DONE]": break
            data = json.loads(chunk)
            if data["choices"][0].get("delta", {}).get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - start
                output_tokens += 1

    return RequestStats(
        ttft=ttft or 0,
        end_to_end=time.perf_counter() - start,
        output_tokens=output_tokens,
        framework=framework,
        scenario=scenario,
    )


async def run_scenario(base_url, model, framework, scenario_name, prompt, sys_prompt, n_requests=20, concurrency=8):
    print(f"\n=== {framework} · {scenario_name} (n={n_requests}, c={concurrency}) ===")
    sem = asyncio.Semaphore(concurrency)

    async def gated(session):
        async with sem:
            return await one_request(session, base_url, model, prompt, sys_prompt, scenario_name, framework)

    async with aiohttp.ClientSession() as session:
        # warmup
        await one_request(session, base_url, model, prompt, sys_prompt, scenario_name, framework)
        # benchmark
        start = time.perf_counter()
        results = await asyncio.gather(*[gated(session) for _ in range(n_requests)])
        wall = time.perf_counter() - start

    total_tok = sum(r.output_tokens for r in results)
    ttfts = sorted([r.ttft for r in results])
    p50_ttft = ttfts[len(ttfts) // 2] * 1000
    p99_ttft = ttfts[int(len(ttfts) * 0.99)] * 1000
    throughput = total_tok / wall

    print(f"  throughput: {throughput:6.1f} tok/s")
    print(f"  TTFT p50:   {p50_ttft:6.0f} ms   p99: {p99_ttft:6.0f} ms")
    return results


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--label", default="vllm", help="vllm / sglang / trtllm")
    p.add_argument("--out", default="results.jsonl")
    args = p.parse_args()

    all_results = []
    scenarios = [
        ("short", PROMPT_SHORT, ""),
        ("long", PROMPT_LONG, ""),
        ("multiturn-shared-sys", PROMPT_SHORT, PROMPT_SYS),
    ]
    for name, pr, sp in scenarios:
        results = await run_scenario(args.base_url, args.model, args.label, name, pr, sp)
        all_results.extend(results)

    with open(args.out, "a") as f:
        for r in all_results:
            f.write(json.dumps(asdict(r)) + "\n")
    print(f"\n结果追加到 {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
