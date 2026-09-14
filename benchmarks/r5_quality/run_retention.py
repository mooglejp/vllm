# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run one bounded 32K baseline/candidate content-retention pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_suite(path: Path, manifest_path: Path) -> tuple[dict[str, dict], dict]:
    raw = path.read_bytes()
    manifest = json.loads(manifest_path.read_text())
    if _sha256(raw) != manifest["suite_sha256"]:
        raise ValueError("retention suite SHA-256 mismatch")
    cases = [json.loads(line) for line in raw.splitlines() if line]
    if len(cases) != manifest["case_count"] or len(cases) != 9:
        raise ValueError("retention suite must contain exactly nine cases")
    if {case["position"] for case in cases} != set(manifest["positions"]):
        raise ValueError("retention positions are incomplete")
    result = {case["case_id"]: case for case in cases}
    if len(result) != len(cases):
        raise ValueError("retention suite has duplicate case IDs")
    for case in cases:
        if len(case["prompt_token_ids"]) != manifest["target_prompt_tokens"]:
            raise ValueError(f"wrong prompt length for {case['case_id']}")
        if case["max_tokens"] != manifest["max_tokens"]:
            raise ValueError(f"wrong output cap for {case['case_id']}")
        if not (
            0
            <= case["document_token_start"]
            < case["target_token_start"]
            < case["target_token_end"]
            <= case["document_token_end"]
            <= case["question_token_start"]
            < case["question_token_end"]
            <= manifest["target_prompt_tokens"]
        ):
            raise ValueError(f"invalid token intervals for {case['case_id']}")
    return result, manifest


def cache_salt(run_id: str) -> str:
    digest = _sha256(run_id.encode())
    return f"r5-retention-{digest}"


def metrics(base: str) -> list[str]:
    with urlopen(base + "/metrics", timeout=10) as response:
        return [
            line
            for line in response.read().decode().splitlines()
            if line and not line.startswith("#")
        ]


def require_idle(lines: list[str]) -> None:
    for key in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
        values = [
            float(line.rsplit(" ", 1)[1])
            for line in lines
            if line.split("{", 1)[0].split(" ", 1)[0] == key
        ]
        if not values or any(value != 0 for value in values):
            raise RuntimeError(f"server is not idle: {key}")


def idle_metrics(base: str) -> list[str]:
    deadline = time.monotonic() + 30
    while True:
        lines = metrics(base)
        try:
            require_idle(lines)
            return lines
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(1)


def resources() -> dict:
    root = Path("/sys/fs/cgroup")
    events = dict(
        line.split() for line in (root / "memory.events").read_text().splitlines()
    )
    return {
        "time": time.time(),
        "memory_current": int((root / "memory.current").read_text()),
        "memory_events": {key: int(value) for key, value in events.items()},
        "memory_stat": (root / "memory.stat").read_text(),
    }


def check_oom(before: dict, after: dict) -> None:
    for key in ("oom", "oom_kill", "oom_group_kill"):
        if after["memory_events"].get(key, 0) > before["memory_events"].get(key, 0):
            raise RuntimeError(f"new cgroup {key}; no automatic retry")


def _read_hook_stats(prefix: Path, run_id: str) -> dict:
    rows = []
    for path in prefix.parent.glob(prefix.name + ".*"):
        try:
            row = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("run_id") == run_id:
            rows.append(row)
    if len(rows) != 1:
        raise RuntimeError(
            f"missing or ambiguous hook stats for {run_id}: {len(rows)} rows"
        )
    return rows[0]


def _request(
    base_url: str,
    model: str,
    case: dict,
    run_id: str,
    mode: str,
) -> dict:
    sampling = {
        "seed": 1201,
        "temperature": 0,
        "top_p": 1,
        "top_k": -1,
        "ignore_eos": False,
        "add_special_tokens": False,
    }
    body = {
        "model": model,
        "prompt": case["prompt_token_ids"],
        "max_tokens": case["max_tokens"],
        "return_token_ids": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": cache_salt(run_id),
        **sampling,
    }
    request = Request(
        base_url + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    first_token = None
    text_parts: list[str] = []
    token_ids: list[int] = []
    stream_errors: list[dict] = []
    finish_reason = None
    usage = None
    try:
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
                text = str(choice.get("text", ""))
                if ids or text:
                    first_token = first_token or time.monotonic()
                token_ids.extend(ids)
                text_parts.append(text)
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
                if event.get("usage") is not None:
                    usage = event["usage"]
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(
            f"generation request failed for {run_id}: {error}"
        ) from error
    finished = time.monotonic()
    text = "".join(text_parts)
    return {
        "mode": mode,
        "case_id": case["case_id"],
        "run_id": run_id,
        "prompt_sha256": case["prompt_sha256"],
        "prompt_tokens": len(case["prompt_token_ids"]),
        "text": text,
        "output_token_ids": token_ids,
        "finish_reason": finish_reason,
        "elapsed_seconds": finished - started,
        "first_token_seconds": (None if first_token is None else first_token - started),
        "stream_errors": stream_errors,
        "usage": usage,
        "request_sampling": body,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen38-27b-tq-mtp")
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--stats-prefix", type=Path, required=True)
    args = parser.parse_args()
    cases, suite_manifest = load_suite(args.suite, args.manifest)
    if args.output.exists():
        raise FileExistsError(f"output directory already exists: {args.output}")
    args.output.mkdir(parents=True)
    run_manifest = {
        "suite_id": suite_manifest["suite_id"],
        "suite_sha256": suite_manifest["suite_sha256"],
        "run_id": args.run_id,
        "case_count": len(cases),
        "order": "B/C then C/B alternating by case",
        "model": args.model,
        "cache_salt": "unique SHA-256 salt for each mode/case",
        "profiling": False,
        "automatic_retry": False,
        "mode_scope": "existing target continuation attention only",
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n"
    )
    initial = resources()
    max_memory = int((Path("/sys/fs/cgroup") / "memory.max").read_text())
    swap = int((Path("/sys/fs/cgroup") / "memory.swap.max").read_text())
    if max_memory > 16 * 2**30 or swap:
        raise RuntimeError("requires RAM <=16 GiB and swap disabled")

    output_rows = {"baseline": [], "candidate": []}
    for index, case in enumerate(cases.values()):
        order = (
            ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
        )
        for mode in order:
            before_metrics = idle_metrics(args.base_url)
            request_id = f"{args.run_id}-{case['case_id']}-{mode}"
            control = {
                "mode": mode,
                "run_id": request_id,
                # The hook's problem interval is the answer-bearing sentence.
                "problem_token_start": case["target_token_start"],
                "problem_token_end": case["target_token_end"],
                "question_token_start": case["question_token_start"],
                "question_token_end": case["question_token_end"],
            }
            temporary = args.control.with_suffix(".tmp")
            temporary.write_text(json.dumps(control))
            temporary.replace(args.control)
            resources_before = resources()
            started = time.monotonic()
            row = _request(args.base_url, args.model, case, request_id, mode)
            row["client_elapsed_seconds"] = time.monotonic() - started
            row["case"] = {
                key: case[key]
                for key in (
                    "content_id",
                    "position",
                    "identifier",
                    "answer",
                    "document_token_start",
                    "document_token_end",
                    "target_token_start",
                    "target_token_end",
                    "question_token_start",
                    "question_token_end",
                )
            }
            row["metrics_before"] = before_metrics
            row["metrics_after"] = metrics(args.base_url)
            row["resources_before"] = resources_before
            row["resources_after"] = resources()
            row["hook"] = _read_hook_stats(args.stats_prefix, request_id)
            row["coverage_expected"] = (
                row["hook"].get("applied_calls", 0) == 0
                if mode == "baseline"
                else row["hook"].get("applied_calls", 0) > 0
            )
            row["target_overlap_expected"] = (
                mode == "baseline" or row["hook"].get("applied_overlap_calls", 0) > 0
            )
            with (args.output / f"{mode}.jsonl").open("a") as output:
                output.write(json.dumps(row, allow_nan=False) + "\n")
                output.flush()
            output_rows[mode].append(row)
            if row["stream_errors"] or not row["output_token_ids"]:
                raise RuntimeError(
                    f"generation error or empty output for {request_id}; artifact saved"
                )
            if row["finish_reason"] not in {"stop", "length"}:
                raise RuntimeError(
                    f"unexpected finish reason for {request_id}: "
                    f"{row['finish_reason']}; artifact saved"
                )
            check_oom(initial, row["resources_after"])
            print(
                f"{index + 1}/9 {mode} {case['case_id']} "
                f"tokens={len(row['output_token_ids'])} "
                f"finish={row['finish_reason']} "
                f"applied={row['hook'].get('applied_calls', 0)} "
                f"overlap={row['hook'].get('applied_overlap_calls', 0)} "
                f"elapsed={row['elapsed_seconds']:.2f}",
                flush=True,
            )
    final = resources()
    check_oom(initial, final)
    run_manifest["completed"] = True
    run_manifest["coverage"] = {
        "baseline_all_unapplied": all(
            row["coverage_expected"] for row in output_rows["baseline"]
        ),
        "candidate_all_applied": all(
            row["coverage_expected"] for row in output_rows["candidate"]
        ),
        "candidate_all_target_overlap": all(
            row["target_overlap_expected"] for row in output_rows["candidate"]
        ),
    }
    run_manifest["resources_final"] = final
    (args.output / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
