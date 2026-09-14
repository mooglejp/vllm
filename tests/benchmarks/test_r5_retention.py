# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the fixed 32K retention fixture and exact scorer."""

import json
from pathlib import Path

from benchmarks.r5_quality.score_retention import score_text


def test_retention_score_only_strips_outer_whitespace():
    answer = "R7N4-K2P9-V6X1"
    assert score_text(f"  {answer}\n", answer)["exact"]
    assert not score_text(f"The value is {answer}.", answer)["exact"]
    assert score_text(f"The value is {answer}.", answer)["contains_answer"]
    assert not score_text(answer.lower(), answer)["exact"]


def test_retention_score_distinguishes_wrong_empty_and_exact():
    answer = "Q5W8-L1D6-Z9C2"
    assert score_text("", answer)["empty"]
    assert not score_text("WRONG", answer)["exact"]
    assert score_text(answer, answer)["exact"]


def test_checked_in_retention_fixture_has_nine_fixed_cases():
    suite = Path("docs/design/artifacts/gfx1201_r5_retention_32k_20260915.jsonl")
    manifest = Path(
        "docs/design/artifacts/gfx1201_r5_retention_32k_manifest_20260915.json"
    )
    rows = [json.loads(line) for line in suite.read_text().splitlines()]
    data = json.loads(manifest.read_text())
    assert len(rows) == 9
    assert {row["position"] for row in rows} == {"early", "middle", "late"}
    assert {len(row["prompt_token_ids"]) for row in rows} == {32768}
    assert all(row["max_tokens"] == 128 for row in rows)
    assert all(row["target_token_start"] < row["target_token_end"] for row in rows)
    assert data["construction"]["prompt_ids_equal_for_baseline_candidate"]
