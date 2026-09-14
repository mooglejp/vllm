# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the context-qualified R5 suite construction."""

from pathlib import Path

import pytest

from benchmarks.r5_quality.build_context_suite import (
    MAX_TOKENS,
    TARGET_PROMPT_TOKENS,
    build_case,
)
from benchmarks.r5_quality.run_context import load_suite


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        if text == "<|im_start|>user\n":
            return [1, 2, 3]
        if text == "<|im_start|>assistant\n":
            return [1, 4, 3]
        return [100 + index for index, _ in enumerate(text.split())]

    def convert_tokens_to_ids(self, token):
        assert token == "<|im_end|>"
        return 9


def _case(task="gsm8k"):
    return {
        "id": f"{task}/1",
        "task": task,
        "max_tokens": 128,
        "prompt_token_ids": [1, 2, 3, 10, 11, 12, 9, 1, 4, 3],
        "reference": "answer",
    }


def test_build_case_preserves_template_task_and_suffix():
    case = build_case(_case(), FakeTokenizer(), list(range(2048)))
    assert len(case["prompt_token_ids"]) == TARGET_PROMPT_TOKENS
    assert case["prompt_token_ids"][:3] == [1, 2, 3]
    assert case["prompt_token_ids"][3 + 2038 : 3 + 2038 + 3] == [10, 11, 12]
    assert case["prompt_token_ids"][-3:] == [1, 4, 3]
    assert case["reference_doc_token_count"] == 2038
    assert case["problem_token_start"] == 2041
    assert case["problem_token_end"] == 2044
    assert case["max_tokens"] == MAX_TOKENS["gsm8k"]


def test_build_case_rejects_short_reference_document():
    with pytest.raises(ValueError, match="reference document is too short"):
        build_case(_case(), FakeTokenizer(), [1])


def test_build_case_rejects_source_over_target():
    case = _case()
    case["prompt_token_ids"] = [0] * (TARGET_PROMPT_TOKENS + 1)
    with pytest.raises(ValueError, match="already exceeds"):
        build_case(case, FakeTokenizer(), list(range(3000)))


def test_checked_in_context_suite_has_fixed_shape_and_caps():
    suite = Path("docs/design/artifacts/gfx1201_r5_quality_context_20260914.jsonl")
    manifest = Path(
        "docs/design/artifacts/gfx1201_r5_quality_context_manifest_20260914.json"
    )
    cases, data = load_suite(suite, manifest)
    assert len(cases) == 308
    assert {len(case["prompt_token_ids"]) for case in cases} == {2048}
    assert all(case["max_tokens"] == MAX_TOKENS[case["task"]] for case in cases)
    assert all(
        0 <= case["problem_token_start"] < case["problem_token_end"] <= 2048
        for case in cases
    )
    assert data["construction_checks"]["baseline_candidate_prompt_ids_identical"]
