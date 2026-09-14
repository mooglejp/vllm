# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize R5 launch correlations without loading a whole trace into RAM."""

import argparse
import bisect
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def events(path: Path):
    decoder = json.JSONDecoder()
    with gzip.open(path, "rt") as stream:
        buf = ""
        marker = '"traceEvents"'
        while marker not in buf:
            chunk = stream.read(65536)
            if not chunk or len(buf) > 16 * 2**20:
                raise ValueError("Missing traceEvents header")
            buf += chunk
        buf = buf.split(marker, 1)[1]
        while "[" not in buf:
            chunk = stream.read(65536)
            if not chunk or len(buf) > 16 * 2**20:
                raise ValueError("Missing traceEvents array")
            buf += chunk
        buf = buf.split("[", 1)[1]
        while True:
            buf = buf.lstrip(" \t\r\n,")
            if buf.startswith("]"):
                return
            try:
                event, end = decoder.raw_decode(buf)
            except json.JSONDecodeError:
                chunk = stream.read(65536)
                if not chunk or len(buf) > 16 * 2**20:
                    raise ValueError("Truncated trace or oversized event") from None
                buf += chunk
                continue
            yield event
            buf = buf[end:]


def summarize(path: Path) -> dict:
    scopes = defaultdict(list)
    for event in events(path):
        if (
            event.get("name") == "r5_candidate_continuation"
            and event.get("cat") == "user_annotation"
        ):
            scopes[(event["pid"], event["tid"])].append(
                (event["ts"], event["ts"] + event["dur"])
            )
    for intervals in scopes.values():
        intervals.sort()
    starts = {key: [start for start, _ in values] for key, values in scopes.items()}
    correlations = set()
    for event in events(path):
        key = (event.get("pid"), event.get("tid"))
        if key not in scopes or event.get("cat") not in {"hip_runtime", "cuda_runtime"}:
            continue
        ts = event["ts"]
        index = bisect.bisect_right(starts[key], ts) - 1
        if index >= 0 and ts < scopes[key][index][1]:
            correlation = event.get("args", {}).get("correlation")
            if correlation is not None:
                correlations.add(correlation)
    names = Counter()
    for event in events(path):
        if event.get("cat") == "kernel":
            correlation = event.get("args", {}).get("correlation")
            if correlation in correlations:
                names[event["name"]] += 1
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "trace": str(path),
        "sha256": digest,
        "candidate_scope_count": sum(map(len, scopes.values())),
        "runtime_correlation_count": len(correlations),
        "correlated_kernel_names": dict(names),
        "flash_attn_fwd_count": sum(
            n for name, n in names.items() if "attn_fwd" in name
        ),
        "correlated_ck_count": sum(n for name, n in names.items() if "ck_tile" in name),
        "note": (
            "Connection evidence only; profiled times are excluded from TTFT gates."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.trace)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
