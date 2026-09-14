# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check repository-style copies against the original scoring helpers."""

import argparse
import importlib.util
import json
from pathlib import Path

import judge_humaneval
import score


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old_score = load(args.original / "score.py", "old_score")
    old_judge = load(args.original / "judge_humaneval.py", "old_judge")
    count = 0
    for path in args.inputs:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            for name in ("_correct", "_semantic_answer"):
                assert getattr(score, name)(row) == getattr(old_score, name)(row)
            if row["task"] == "humaneval":
                assert score._human_source(row) == old_score._human_source(row)
                assert judge_humaneval._candidate_source(
                    row
                ) == old_judge._candidate_source(row)
            count += 1
    with args.output.open("x") as output:
        json.dump({"rows_checked": count, "identical": True}, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
