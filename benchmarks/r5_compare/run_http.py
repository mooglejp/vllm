# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a fixed suite against one OpenAI-compatible serving endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _get(url: str) -> dict | list | str:
    try:
        with urlopen(url, timeout=10) as response:
            raw = response.read().decode(errors="replace")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw[:20000]
    except (HTTPError, URLError, TimeoutError) as error:
        return {"error": str(error)}


def resource_snapshot() -> dict[str, str]:
    result = {}
    for name, command in (
        (
            "rocm_smi",
            ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--showtemp", "--csv"],
        ),
        (
            "nvidia_smi",
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader",
            ],
        ),
    ):
        try:
            result[name] = subprocess.run(
                command, capture_output=True, text=True, timeout=10, check=False
            ).stdout.strip()
        except (OSError, subprocess.TimeoutExpired) as error:
            result[name] = f"unavailable: {error}"
    return result


def _write_control(args: argparse.Namespace, run_id: str, prompt_tokens: int) -> None:
    if not args.hook_control_host:
        return
    path = Path(args.hook_control_host)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "mode": args.hook_mode,
                "run_id": run_id,
                "problem_token_start": 0,
                "problem_token_end": max(prompt_tokens, 1) * 2,
                "question_token_start": 0,
                "question_token_end": max(prompt_tokens, 1) * 2,
            }
        )
    )
    temporary.replace(path)


def _hook_stats(args: argparse.Namespace, run_id: str) -> dict | None:
    if not args.hook_stats_host:
        return None
    prefix = Path(args.hook_stats_host)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        for path in prefix.parent.glob(prefix.name + ".*"):
            try:
                row = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if row.get("run_id") == run_id:
                return row
        time.sleep(0.2)
    return {"error": "hook stats not observed", "run_id": run_id}


def request(
    args: argparse.Namespace,
    row: dict,
    sample_id: str,
    warmup: bool = False,
) -> dict:
    run_id = f"{args.run_id}/{row['id']}/{sample_id}"
    prompt_tokens = int(row.get("target_prompt_tokens", 0))
    _write_control(args, run_id, prompt_tokens)
    body = {
        "model": args.model,
        "messages": row["messages"],
        "temperature": 0,
        "top_p": 1,
        "seed": 1201,
        "max_tokens": row["max_tokens"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": args.kind == "speed",
    }
    if args.engine == "vllm":
        body["cache_salt"] = f"r5-compare-{run_id}"
        body["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        body["cache_prompt"] = False
    resource_before = resource_snapshot()
    started = time.perf_counter()
    first_content = None
    pieces: list[str] = []
    finish_reason = None
    usage = None
    stream_error = None
    request_error = None
    try:
        request_obj = Request(
            args.base_url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request_obj, timeout=args.timeout) as response:
            for raw_line in response:
                line = raw_line.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                if "error" in event:
                    stream_error = event["error"]
                    continue
                if event.get("usage") is not None:
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    first_content = first_content or time.perf_counter()
                    pieces.append(piece)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        request_error = str(error)
    finished = time.perf_counter()
    text = "".join(pieces)
    output_tokens = None
    if usage:
        output_tokens = usage.get("completion_tokens")
    generation_seconds = None if first_content is None else finished - first_content
    result = {
        "run_id": run_id,
        "engine": args.engine,
        "suite_kind": args.kind,
        "case_id": row["id"],
        "sample_id": sample_id,
        "warmup": warmup,
        "target_prompt_tokens": row.get("target_prompt_tokens"),
        "text": text,
        "finish_reason": finish_reason,
        "output_tokens": output_tokens,
        "usage": usage,
        "elapsed_seconds": finished - started,
        "ttft_seconds": None if first_content is None else first_content - started,
        "generation_seconds": generation_seconds,
        "tokens_per_second": (
            None
            if output_tokens is None or output_tokens <= 1 or generation_seconds is None
            else (output_tokens - 1) / generation_seconds
        ),
        "request_error": request_error,
        "stream_error": stream_error,
        "prompt_text_sha256": hashlib.sha256(row["prompt_text"].encode()).hexdigest(),
        "resources_before": resource_before,
        "endpoint_health": _get(args.base_url.rstrip("/") + "/health"),
        "endpoint_slots": _get(args.base_url.rstrip("/") + "/slots"),
        "endpoint_metrics": _get(args.base_url.rstrip("/") + "/metrics"),
    }
    result["hook"] = _hook_stats(args, run_id)
    result["resources_after"] = resource_snapshot()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument(
        "--kind", choices=("speed", "quality", "retention"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine", choices=("vllm", "llama"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--hook-control-host")
    parser.add_argument("--hook-stats-host")
    parser.add_argument(
        "--hook-mode", choices=("baseline", "candidate"), default="candidate"
    )
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.suite.read_text().splitlines() if line]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    warmup_path = args.output.with_name(args.output.stem + ".warmup.jsonl")
    output_rows = []
    with args.output.open("w") as output:
        for row in rows:
            if args.warmup:
                warm = request(args, row, "warmup", warmup=True)
                with warmup_path.open("a") as warmup_output:
                    warmup_output.write(json.dumps(warm, ensure_ascii=False) + "\n")
            count = args.samples if args.kind == "speed" else 1
            for index in range(count):
                result = request(args, row, f"sample-{index + 1}")
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
                output_rows.append(result)
                print(
                    f"{args.engine} {args.kind} {row['id']} sample={index + 1} "
                    f"tokens={result['output_tokens']} "
                    f"finish={result['finish_reason']} "
                    f"ttft={result['ttft_seconds']}",
                    flush=True,
                )
    summary = {
        "engine": args.engine,
        "kind": args.kind,
        "rows": len(output_rows),
        "request_errors": sum(row["request_error"] is not None for row in output_rows),
        "stream_errors": sum(row["stream_error"] is not None for row in output_rows),
        "finish_reasons": {
            reason: sum(row["finish_reason"] == reason for row in output_rows)
            for reason in sorted({row["finish_reason"] for row in output_rows})
        },
    }
    args.output.with_name(args.output.stem + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
