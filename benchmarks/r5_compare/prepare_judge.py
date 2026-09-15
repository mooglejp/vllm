# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Join a fixed suite's references onto cross-runtime response records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    suite = {
        row["id"]: row
        for row in (json.loads(line) for line in args.suite.read_text().splitlines())
    }
    responses = [
        json.loads(line) for line in args.responses.read_text().splitlines() if line
    ]
    output = []
    for response in responses:
        case_id = response.get("case_id", response.get("id"))
        if case_id not in suite:
            raise ValueError(f"response case is not in suite: {case_id}")
        merged = dict(suite[case_id])
        merged.update(response)
        merged["id"] = case_id
        output.append(merged)
    if len({row["id"] for row in output}) != len(output):
        raise ValueError("duplicate response case IDs")
    expected = set(suite)
    actual = {row["id"] for row in output}
    if actual != expected:
        raise ValueError(f"suite/response mismatch: missing={expected - actual}")
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output)
    )


if __name__ == "__main__":
    main()
