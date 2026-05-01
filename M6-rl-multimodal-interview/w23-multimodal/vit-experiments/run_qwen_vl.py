"""
W23 任务：跑通 Qwen2.5-VL，给一张图让模型描述

前置:
    pip install vllm>=0.5.0
    pip install pillow

启动 vllm server（另一个终端）:
    vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000

然后跑这个脚本:
    python run_qwen_vl.py --image https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png
"""

import argparse
import base64
from openai import OpenAI


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="image URL or local file path")
    p.add_argument("--question", default="详细描述这张图")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    args = p.parse_args()

    # 如果是本地文件，转 base64
    if args.image.startswith("http"):
        image_url = args.image
    else:
        with open(args.image, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        image_url = f"data:image/png;base64,{b64}"

    client = OpenAI(base_url=args.base_url, api_key="dummy")

    resp = client.chat.completions.create(
        model=args.model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": args.question},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }],
        max_tokens=512,
    )

    print("=== Qwen2.5-VL 回答 ===")
    print(resp.choices[0].message.content)
    print()
    print(f"Prompt tokens: {resp.usage.prompt_tokens}")
    print(f"Output tokens: {resp.usage.completion_tokens}")


if __name__ == "__main__":
    main()
