# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the fixed 2,048-token context-qualified R5 quality A/B once."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

TASK_COUNTS = {"gsm8k": 64, "mmlu": 80, "humaneval": 164}
MAX_TOKENS = {"gsm8k": 1024, "mmlu": 64, "humaneval": 2048}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_suite(path: Path, manifest_path: Path) -> tuple[list[dict], dict]:
    raw = path.read_bytes()
    manifest = json.loads(manifest_path.read_text())
    if _sha256(raw) != manifest["suite_sha256"]:
        raise ValueError("context suite SHA-256 mismatch")
    cases = [json.loads(line) for line in raw.splitlines()]
    if Counter(case["task"] for case in cases) != Counter(TASK_COUNTS):
        raise ValueError("context suite task counts are incomplete")
    if len({case["id"] for case in cases}) != 308:
        raise ValueError("context suite has duplicate or missing IDs")
    for case in cases:
        if len(case["prompt_token_ids"]) != manifest["target_prompt_tokens"]:
            raise ValueError(f"wrong prompt length for {case['id']}")
        if case["max_tokens"] != MAX_TOKENS[case["task"]]:
            raise ValueError(f"wrong output cap for {case['id']}")
        if case["problem_token_start"] >= case["problem_token_end"]:
            raise ValueError(f"invalid problem positions for {case['id']}")
    return cases, manifest


def cache_salt(run_id: str) -> str:
    value = "r5-context-quality-" + _sha256(run_id.encode())
    if len(value) > 128 or any(char in value for char in "@/\\\x00"):
        raise ValueError("invalid cache salt")
    return value


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


def read_scope(prefix: Path, run_id: str, mode: str, case: dict) -> dict:
    rows = [
        json.loads(path.read_text()) for path in prefix.parent.glob(prefix.name + ".*")
    ]
    rows = [row for row in rows if row.get("run_id") == run_id]
    if len(rows) != 1:
        raise RuntimeError("missing or ambiguous per-request hook counters")
    row = rows[0]
    if row.get("problem_token_start") != case["problem_token_start"]:
        raise RuntimeError("hook problem start does not match suite")
    if row.get("problem_token_end") != case["problem_token_end"]:
        raise RuntimeError("hook problem end does not match suite")
    expected = row["calls"] if mode == "candidate" else 0
    if row["applied_calls"] != expected:
        raise RuntimeError("unexpected attention application count")
    row["coverage"] = "applied" if row["applied_calls"] else "unapplied_regression"
    row["problem_overlap"] = bool(row.get("overlap_calls"))
    row["applied_problem_overlap"] = bool(row.get("applied_overlap_calls"))
    if mode == "candidate" and not row["applied_problem_overlap"]:
        raise RuntimeError("candidate did not apply to a problem-overlapping chunk")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--control", type=Path, default=Path("/dev/shm/r5_model_control.json")
    )
    parser.add_argument("--stats", type=Path, required=True)
    args = parser.parse_args()
    cases, suite_manifest = load_suite(args.suite, args.manifest)
    args.output.mkdir(exist_ok=False, parents=True)
    root = Path("/sys/fs/cgroup")
    if int((root / "memory.max").read_text()) > 16 * 2**30 or int(
        (root / "memory.swap.max").read_text()
    ):
        raise RuntimeError("requires RAM <=16 GiB and swap disabled")
    manifest = {
        "suite_id": suite_manifest["suite_id"],
        "suite_sha256": suite_manifest["suite_sha256"],
        "run_id": args.run_id,
        "counts": TASK_COUNTS,
        "prompt_tokens": 2048,
        "max_tokens": MAX_TOKENS,
        "sampling": suite_manifest["sampling"],
        "order": "alternating baseline/candidate then candidate/baseline per case",
        "salt": "unique SHA-256 salt for every run/mode/case; no arm sharing",
        "coverage": (
            "candidate applied on every case and at least one applied chunk overlaps "
            "the original problem token interval"
        ),
        "historical_results_preserved": True,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    initial = resources()
    for index, case in enumerate(cases):
        if case["max_tokens"] != MAX_TOKENS[case["task"]]:
            raise RuntimeError(f"manifest/case output cap mismatch: {case['id']}")
        for mode in (
            ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
        ):
            before = idle_metrics(args.base_url)
            run_id = f"{args.run_id}/{mode}/{case['id']}"
            temporary = args.control.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "mode": mode,
                        "run_id": run_id,
                        "problem_token_start": case["problem_token_start"],
                        "problem_token_end": case["problem_token_end"],
                    }
                )
            )
            temporary.replace(args.control)
            body = dict(
                suite_manifest["sampling"],
                model="qwen38-27b-tq-mtp",
                prompt=case["prompt_token_ids"],
                max_tokens=case["max_tokens"],
                return_token_ids=True,
                cache_salt=cache_salt(run_id),
                logprobs=1,
            )
            request = Request(
                args.base_url + "/v1/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            started = time.monotonic()
            try:
                with urlopen(request, timeout=1800) as response:
                    result = json.load(response)
            except HTTPError as error:
                (args.output / "http-error.json").write_text(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "status": error.code,
                            "body": error.read().decode(errors="replace"),
                        },
                        indent=2,
                    )
                )
                raise
            elapsed = time.monotonic() - started
            after = metrics(args.base_url)
            resource = resources()
            choice = result["choices"][0]
            logprobs = (choice.get("logprobs") or {}).get("token_logprobs") or []
            finite = bool(logprobs) and all(
                value is not None and math.isfinite(value) for value in logprobs
            )
            hook = read_scope(args.stats, run_id, mode, case)
            record = dict(
                case,
                mode=mode,
                run_id=run_id,
                request_max_tokens=body["max_tokens"],
                request_sampling=body,
                response=result,
                text=choice["text"],
                output_token_ids=choice["token_ids"],
                finish_reason=choice["finish_reason"],
                elapsed_seconds=elapsed,
                metrics_before=before,
                metrics_after=after,
                resources=resource,
                hook=hook,
                finite_logprobs=finite,
                empty=not choice["text"].strip(),
                truncated=choice["finish_reason"] == "length",
                prompt_sha256=_sha256(
                    json.dumps(case["prompt_token_ids"], separators=(",", ":")).encode()
                ),
            )
            with (args.output / f"{mode}.jsonl").open("a") as output:
                output.write(json.dumps(record, allow_nan=False) + "\n")
                output.flush()
            print(
                f"{index + 1}/308 {mode} {case['id']} "
                f"tokens={len(choice['token_ids'])} "
                f"applied={hook['applied_calls']} "
                f"overlap={hook.get('applied_overlap_calls', 0)} "
                f"elapsed={elapsed:.2f}",
                flush=True,
            )
            check_oom(initial, resource)
            if (
                not finite
                or not choice["token_ids"]
                or choice["finish_reason"] not in {"stop", "length"}
            ):
                raise RuntimeError("generation/finite gate failed; result preserved")


if __name__ == "__main__":
    main()
