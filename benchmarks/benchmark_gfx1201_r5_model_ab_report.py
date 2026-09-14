# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate and aggregate the fixed five-pair R5 model A/B matrix."""

import argparse
import json
import statistics
from pathlib import Path


def summarize(directory: Path) -> dict:
    records = []
    hashes = set()
    salts = set()
    samples = {"baseline": [], "candidate": []}
    for index in range(5):
        order = (
            ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
        )
        for mode in order:
            path = directory / f"sample{index}-{mode}.json"
            record = json.loads(path.read_text())
            result = record["requests"][0]["result"]
            assert record["mode"] == mode and not record["profiled"]
            assert result["prompt_tokens"] == 32768
            assert len(result["completion_token_ids"]) == 64
            assert result["usage"]["completion_tokens"] == 64
            assert result["cache_salt"] not in salts
            salts.add(result["cache_salt"])
            hashes.add(result["prompt_sha256"])
            (stats,) = record["hook_stats"]
            assert stats["run_id"] == record["run_id"]
            assert len(stats["layers"]) == 16 and stats["calls"] > 0
            assert stats["applied_calls"] == (
                stats["calls"] if mode == "candidate" else 0
            )
            assert 0 < result["first_token_s"] < result["elapsed_s"]
            samples[mode].append(result["first_token_s"])
            records.append({"file": path.name, "record": record})
    assert len(hashes) == 1
    medians = {mode: statistics.median(values) for mode, values in samples.items()}
    reduction = 1 - medians["candidate"] / medians["baseline"]
    return {
        "status": (
            "model_ttft_passed_quality_and_operational_evaluation_pending"
            if reduction >= 0.10
            else "model_ttft_gate_failed"
        ),
        "ttft_samples_s": samples,
        "ttft_median_s": medians,
        "ttft_reduction_fraction": reduction,
        "ttft_speedup": medians["baseline"] / medians["candidate"],
        "ttft_gate": {"minimum_reduction_fraction": 0.10, "pass": reduction >= 0.10},
        "measurement": (
            "same-process, same-load; alternating arms; salted cold prefixes"
        ),
        "memory_note": (
            "hook peaks are process-lifetime allocator peaks, including startup"
        ),
        "production_adopted": False,
        "quality_note": "Greedy token differences are recorded, not a quality verdict.",
        "raw_records": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.directory)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "raw_records"}, indent=2))
