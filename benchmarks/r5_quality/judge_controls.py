# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Demonstrate that isolated HumanEval grading actually executes stdin."""

import argparse
import json
from pathlib import Path

from judge_humaneval import _judge
from verify_scorers import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = load(args.original, "original_judge")
    venv = Path("/home/emmett/vllm-tq/.venv")
    python_root = Path(
        "/home/emmett/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu"
    )
    records = []
    for expected in (0, 1):
        row = {
            "id": f"returns-zero-expect-{expected}",
            "text": "def answer():\n    return 0\n",
            "reference": {
                "prompt": "def answer():\n    pass\n",
                "entry_point": "answer",
                "test": "def check(candidate):\n"
                f"    assert candidate() == {expected}\n",
            },
        }
        for name, judge in (("original_no_stdin", original._judge), ("fixed", _judge)):
            result = judge(row, args.image, venv, python_root)
            records.append(dict(result, judge=name, expected_pass=expected == 0))
    with args.output.open("x") as output:
        json.dump({"records": records}, output, indent=2)
        output.write("\n")
    assert all(
        r["passed"] == r["expected_pass"] for r in records if r["judge"] == "fixed"
    )
    assert any(
        r["passed"] != r["expected_pass"]
        for r in records
        if r["judge"] == "original_no_stdin"
    )


if __name__ == "__main__":
    main()
