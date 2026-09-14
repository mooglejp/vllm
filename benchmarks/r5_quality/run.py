# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequential, disk-backed R5 quality A/B; never changes production dispatch."""

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

SUITE_SHA256 = "8476e86f75bc2b08a19f187e6ca11bef4af161bed203a0817c39a50b1150338a"


def load_suite(path):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SUITE_SHA256:
        raise ValueError("Fixed quality suite digest mismatch")
    cases = [json.loads(line) for line in data.splitlines()]
    if (
        Counter(c["task"] for c in cases) != {"gsm8k": 64, "mmlu": 80, "humaneval": 164}
        or len({c["id"] for c in cases}) != 308
    ):
        raise ValueError("Incomplete or duplicate quality cases")
    return cases


def cache_salt(run_id):
    return "r5-quality-" + hashlib.sha256(run_id.encode()).hexdigest()


def metrics(base):
    with urlopen(base + "/metrics", timeout=10) as response:
        return [
            line
            for line in response.read().decode().splitlines()
            if line and not line.startswith("#")
        ]


def require_idle(lines):
    for key in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
        values = [
            float(line.rsplit(" ", 1)[1])
            for line in lines
            if line.split("{", 1)[0].split(" ", 1)[0] == key
        ]
        if not values or any(value != 0 for value in values):
            raise RuntimeError(f"Server is not demonstrably idle: {key}")


def idle_metrics(base):
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


def resources():
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


def check_oom(before, after):
    for key in ("oom", "oom_kill", "oom_group_kill"):
        if after["memory_events"].get(key, 0) > before["memory_events"].get(key, 0):
            raise RuntimeError(f"New cgroup {key}; no automatic retry")


def scope(prefix, run_id, mode):
    rows = [json.loads(p.read_text()) for p in prefix.parent.glob(prefix.name + ".*")]
    rows = [row for row in rows if row.get("run_id") == run_id]
    if len(rows) != 1:
        raise RuntimeError("Missing or ambiguous per-request hook counters")
    row = rows[0]
    expected = row["calls"] if mode == "candidate" else 0
    if row["applied_calls"] != expected:
        raise RuntimeError("Unexpected attention application count")
    row["coverage"] = "applied" if row["applied_calls"] else "unapplied_regression"
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--control", type=Path, default=Path("/dev/shm/r5_model_control.json")
    )
    parser.add_argument("--stats", type=Path, required=True)
    args = parser.parse_args()
    cases = load_suite(args.suite)
    args.output.mkdir(exist_ok=False, parents=True)
    root = Path("/sys/fs/cgroup")
    if int((root / "memory.max").read_text()) > 16 * 2**30 or int(
        (root / "memory.swap.max").read_text()
    ):
        raise RuntimeError("Requires RAM <=16 GiB, swap disabled")
    manifest = {
        "base": "d5fec6675f832ba6bf753247acd092ef86e5cd56",
        "suite_sha256": SUITE_SHA256,
        "run_id": args.run_id,
        "counts": dict(Counter(c["task"] for c in cases)),
        "prompts": "exact stored token IDs, original chat template already applied",
        "sampling": {
            "seed": 1201,
            "temperature": 0,
            "top_p": 1,
            "top_k": -1,
            "ignore_eos": False,
            "add_special_tokens": False,
        },
        "max_tokens": {"gsm8k": 128, "mmlu": 16, "humaneval": 384},
        "order": "alternating B/C then C/B per case; sequential, one model load",
        "salt": "r5-quality- + SHA256(run_id/mode/case_id); no inter-arm sharing",
        "gate": "per-task correct count >= paired MTP2 baseline; no new errors",
        "scoring": "unchanged preserved score.py and isolated judge_humaneval.py",
        "nonfinite": "token log probabilities; not exhaustive hidden-state checks",
        "historical_gates": "TTFT pass and strict Math diagnostic failure unchanged",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    initial = resources()
    for index, case in enumerate(cases):
        for mode in (
            ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
        ):
            before = idle_metrics(args.base_url)
            run_id = f"{args.run_id}/{mode}/{case['id']}"
            temporary = args.control.with_suffix(".tmp")
            temporary.write_text(json.dumps({"mode": mode, "run_id": run_id}))
            temporary.replace(args.control)
            body = dict(
                manifest["sampling"],
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
                with urlopen(request, timeout=900) as response:
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
                v is not None and math.isfinite(v) for v in logprobs
            )
            record = dict(
                case,
                mode=mode,
                run_id=run_id,
                response=result,
                text=choice["text"],
                output_token_ids=choice["token_ids"],
                finish_reason=choice["finish_reason"],
                elapsed_seconds=elapsed,
                metrics_before=before,
                metrics_after=after,
                resources=resource,
                hook=scope(args.stats, run_id, mode),
                finite_logprobs=finite,
                empty=not choice["text"].strip(),
                truncated=choice["finish_reason"] == "length",
            )
            with (args.output / f"{mode}.jsonl").open("a") as output:
                output.write(json.dumps(record, allow_nan=False) + "\n")
                output.flush()
            print(
                f"{index + 1}/308 {mode} {case['id']} "
                f"tokens={len(choice['token_ids'])} "
                f"applied={record['hook']['applied_calls']} elapsed={elapsed:.2f}",
                flush=True,
            )
            check_oom(initial, resource)
            if (
                not finite
                or not choice["token_ids"]
                or choice["finish_reason"] not in {"stop", "length"}
            ):
                raise RuntimeError(
                    "Generation/finite gate failed; result preserved, no retry"
                )


if __name__ == "__main__":
    main()
