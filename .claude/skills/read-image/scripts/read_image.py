#!/usr/bin/env python3
"""Analyze an image using Xiaomi MiMo-V2.5 Vision API.

Usage:
    python read_image.py <image_path> [options]

The API is OpenAI-compatible. Images are base64-encoded and sent as
image_url content parts alongside the text prompt.
"""

import argparse
import base64
import json
import mimetypes
import os
import sys

import requests
from dotenv import load_dotenv

# Fix Unicode output on Windows consoles that default to GBK
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

API_KEY = os.getenv("xiaomi_api_key")
if not API_KEY:
    print("Error: xiaomi_api_key not found in .env file")
    sys.exit(1)

API_URL = "https://token-plan-cn.xiaomimimo.com/v1/chat/completions"
DEFAULT_MODEL = "mimo-v2.5"


def encode_image(image_path: str) -> str:
    """Read an image file and return a base64-encoded data URL string."""
    if not os.path.isfile(image_path):
        print(f"Error: file not found: {image_path}")
        sys.exit(1)

    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None or not mime_type.startswith("image/"):
        print(f"Warning: could not determine MIME type for {image_path}, falling back to image/png")
        mime_type = "image/png"

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{image_data}"


def build_payload(
    image_path: str,
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float | None,
    system_prompt: str | None,
    stream: bool,
) -> dict:
    """Build the request payload for the MiMo API."""
    data_url = encode_image(image_path)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    })

    payload: dict = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "stream": stream,
    }
    if temperature is not None:
        payload["temperature"] = temperature

    return payload


def _handle_stream(headers: dict, payload: dict) -> str:
    """Handle SSE streaming response, printing tokens as they arrive."""
    full_content = ""
    with requests.post(
        API_URL, headers=headers, json=payload, timeout=120, stream=True
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    print(content, end="", flush=True)
                    full_content += content
            if "usage" in chunk:
                _print_usage(chunk["usage"])
        print()
    return full_content


def _print_usage(usage: dict) -> None:
    """Print token usage summary."""
    prompt_tokens = usage.get("prompt_tokens", "?")
    completion_tokens = usage.get("completion_tokens", "?")
    total_tokens = usage.get("total_tokens", "?")
    image_tokens = (
        usage.get("prompt_tokens_details", {}).get("image_tokens", 0) or 0
    )
    print(
        f"\n[Tokens: prompt={prompt_tokens} (img={image_tokens}), "
        f"completion={completion_tokens}, total={total_tokens}]"
    )


def call_mimo(
    image_path: str,
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    temperature: float | None = None,
    system_prompt: str | None = None,
    stream: bool = False,
) -> str:
    """Send an image to the MiMo API and return the model's response."""
    payload = build_payload(
        image_path, prompt, model, max_tokens, temperature, system_prompt, stream
    )
    headers = {
        "api-key": API_KEY,
        "Content-Type": "application/json",
    }

    if stream:
        return _handle_stream(headers, payload)

    resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    if not resp.ok:
        print(f"Error: API returned {resp.status_code}\n{resp.text}")
        sys.exit(1)

    data = resp.json()

    if "choices" in data and len(data["choices"]) > 0:
        message = data["choices"][0]["message"]
        content = message.get("content", "")
        # Print reasoning if present (thinking mode)
        if message.get("reasoning_content"):
            print(f"[Reasoning: {message['reasoning_content'][:200]}...]")
        if "usage" in data:
            _print_usage(data["usage"])
        return content
    else:
        return json.dumps(data, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze an image using Xiaomi MiMo-V2.5 Vision API"
    )
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument(
        "--prompt", "-p",
        default="请详细描述这张图片的内容。",
        help="Question or instruction about the image",
    )
    parser.add_argument(
        "--stream", "-s",
        action="store_true",
        help="Enable SSE streaming output",
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Model ID to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=None,
        help="Sampling temperature (0–1.5). Omit to use model default.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum output tokens (default: 4096)",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="System message to set model behavior",
    )
    args = parser.parse_args()

    result = call_mimo(
        image_path=args.image,
        prompt=args.prompt,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        system_prompt=args.system_prompt,
        stream=args.stream,
    )
    if not args.stream:
        print(result)


if __name__ == "__main__":
    main()
