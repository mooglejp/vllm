# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Send the fixed-token R5 4K smoke and cold 32K requests to a local server."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


def _wait_health(base_url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urlopen(f"{base_url}/health", timeout=2):
                return
        except (OSError, URLError):
            time.sleep(1)
    raise TimeoutError(f"server did not become healthy: {base_url}")


def _request(
    base_url: str,
    model: str,
    prompt: list[int],
    cache_salt: str,
    max_tokens: int,
) -> dict[str, Any]:
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        "cache_salt": cache_salt,
    }
    request = Request(
        f"{base_url}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    first = None
    text_parts: list[str] = []
    token_ids: list[int] = []
    stream_errors: list[dict[str, Any]] = []
    usage: dict[str, Any] | None = None
    with urlopen(request, timeout=1800) as response:
        for raw_line in response:
            line = raw_line.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            if "error" in event:
                stream_errors.append(event)
                continue
            choices = event.get("choices", [])
            choice = choices[0] if choices else {}
            ids = choice.get("token_ids") or []
            if ids or choice.get("text"):
                first = first or time.monotonic()
            token_ids.extend(ids)
            text_parts.append(str(choice.get("text", "")))
            if event.get("usage") is not None:
                usage = event["usage"]
    finished = time.monotonic()
    text = "".join(text_parts)
    return {
        "prompt_tokens": len(prompt),
        "prompt_sha256": hashlib.sha256(
            json.dumps(prompt, separators=(",", ":")).encode()
        ).hexdigest(),
        "cache_salt": cache_salt,
        "first_token_s": None if first is None else first - started,
        "elapsed_s": finished - started,
        "completion_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "completion_text_length": len(text),
        "completion_token_ids": token_ids,
        "stream_errors": stream_errors,
        "usage": usage,
    }


def _metrics(base_url: str) -> list[str]:
    with urlopen(f"{base_url}/metrics", timeout=10) as response:
        lines = response.read().decode().splitlines()
    return [
        line
        for line in lines
        if not line.startswith("#")
        and any(key in line for key in ("spec_decode", "prefix_cache", "kv_cache"))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen38-27b-tq-mtp")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--health-timeout-s", type=float, default=900)
    parser.add_argument(
        "--case", choices=("smoke4k", "cold32k", "both"), default="both"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("baseline", "candidate"))
    parser.add_argument("--control-file", type=Path)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--hook-stats-prefix", default="/dev/shm/r5_model_stats.json")
    args = parser.parse_args()

    prompts = json.loads(args.prompt_file.read_text())
    _wait_health(args.base_url, args.health_timeout_s)
    if args.control_file:
        if not args.mode:
            parser.error("--control-file requires --mode")
        args.control_file.write_text(
            json.dumps({"mode": args.mode, "run_id": args.run_id})
        )
    rows = []
    metrics_before = _metrics(args.base_url)
    if args.profile:
        with urlopen(
            Request(f"{args.base_url}/start_profile", method="POST"), timeout=180
        ):
            pass
    for label in ("smoke4k", "cold32k") if args.case == "both" else (args.case,):
        prompt = prompts[label]
        print(f"request {label} tokens={len(prompt)}", flush=True)
        rows.append(
            {
                "label": label,
                "result": _request(
                    args.base_url,
                    args.model,
                    prompt,
                    cache_salt=f"r5-{label}-{args.run_id}",
                    max_tokens=args.max_tokens,
                ),
            }
        )
        print(json.dumps(rows[-1], sort_keys=True), flush=True)
    report = {
        "mode": args.mode,
        "run_id": args.run_id,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "same_token_id_prompt": True,
        "requests": rows,
        "metrics_before": metrics_before,
        "metrics_after": _metrics(args.base_url),
        "profiled": args.profile,
        "hook_stats": [
            json.loads(path.read_text())
            for path in Path(args.hook_stats_prefix).parent.glob(
                Path(args.hook_stats_prefix).name + ".*"
            )
            if json.loads(path.read_text()).get("run_id") == args.run_id
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for row in rows:
        result = row["result"]
        if (
            result["stream_errors"]
            or len(result["completion_token_ids"]) != args.max_tokens
        ):
            raise RuntimeError(
                "Generation failed; incomplete response saved in artifact"
            )
    if args.profile:
        with urlopen(
            Request(f"{args.base_url}/stop_profile", method="POST"), timeout=180
        ):
            pass
    if args.control_file:
        stats = report["hook_stats"]
        if len(stats) != 1 or not stats[0]["calls"]:
            raise RuntimeError("No unique target continuation scope record")
        expected = stats[0]["calls"] if args.mode == "candidate" else 0
        if stats[0]["applied_calls"] != expected or len(stats[0]["layers"]) != 16:
            raise RuntimeError("Unexpected target attention scope or application count")


if __name__ == "__main__":
    main()
