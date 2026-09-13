# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a fixed greedy long-prefill request against a local OpenAI server.

This helper is intentionally an opt-in model-level A/B harness for the
gfx1201 unified q128 continuation experiment.  It does not set any server
environment variables; launch the baseline and candidate servers separately
with identical arguments and only change the candidate opt-in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

from transformers import AutoTokenizer


def _prompt(tokenizer, target_tokens: int) -> list[int]:
    template = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": (
                    "Notes:\n{CONTEXT}\n\nExplain how prompt-prefix caching "
                    "changes prefill and decode work in a language-model server. "
                    "Give a concrete example and mention its limitations."
                ),
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    start, end = template.split("{CONTEXT}")
    prefix = tokenizer.encode(start, add_special_tokens=False)
    suffix = tokenizer.encode(end, add_special_tokens=False)
    notes = tokenizer.encode(
        "A team serves a language model to many users. Requests can share a "
        "system prompt. The server retains key and value tensors for completed "
        "prompt blocks. The team measures time to first token and generated "
        "tokens per second separately. GPU memory capacity limits retained "
        "blocks.\n",
        add_special_tokens=False,
    )
    count = target_tokens - len(prefix) - len(suffix)
    if count < 0:
        raise ValueError(f"target prompt is too short: {target_tokens}")
    return prefix + (notes * (count // len(notes) + 1))[:count] + suffix


def _request(base_url: str, model: str, prompt: list[int], output_tokens: int):
    body = {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "top_p": 1,
        "top_k": -1,
        "max_tokens": output_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        "cache_salt": f"gfx1201-unified-{time.time_ns()}",
    }
    request = Request(
        base_url + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    tokens: list[int] = []
    usage = None
    metrics = None
    first_time = last_time = None
    first_chunk_tokens = 0
    started = time.perf_counter()
    with urlopen(request, timeout=900) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            event = json.loads(payload)
            if "error" in event:
                raise RuntimeError(event)
            for choice in event.get("choices", []):
                delta = choice.get("token_ids") or []
                if delta:
                    last_time = time.perf_counter()
                    if first_time is None:
                        first_time = last_time
                        first_chunk_tokens = len(delta)
                    tokens.extend(delta)
            if event.get("usage"):
                usage = event["usage"]
            if event.get("metrics"):
                metrics = event["metrics"]
    elapsed = time.perf_counter() - started
    if usage is None or len(tokens) != usage["completion_tokens"]:
        raise RuntimeError(f"inconsistent response: usage={usage} tokens={len(tokens)}")
    spec = (metrics or {}).get("speculative_decoding")
    return {
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": len(tokens),
        "elapsed_s": elapsed,
        "e2e_tokens_s": len(tokens) / elapsed,
        "ttft_s": first_time - started,
        "decode_tokens_s": (
            (len(tokens) - first_chunk_tokens) / (last_time - first_time)
            if last_time is not None and last_time > first_time
            else None
        ),
        "speculative_decoding": spec,
        "accepted_draft_tokens_per_step": (
            spec["num_accepted_draft_tokens"] / spec["num_spec_steps"]
            if spec and spec["num_spec_steps"]
            else None
        ),
        "token_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        "output_token_ids": tokens,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default="/model")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen38-27b-tq-mtp")
    parser.add_argument("--contexts", type=int, nargs="+", default=[4096, 32768])
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for context in args.contexts:
        prompt = _prompt(tokenizer, context)
        prompt_hash = hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
        for repetition in range(args.repeats + 1):
            result = _request(args.base_url, args.model, prompt, args.output_tokens)
            result.update(
                label=args.label,
                repetition=repetition,
                phase="warmup" if repetition == 0 else "measured",
                prompt_sha256=prompt_hash,
                prompt_target_tokens=context,
            )
            with args.output.open("a") as output_file:
                output_file.write(json.dumps(result) + "\n")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
