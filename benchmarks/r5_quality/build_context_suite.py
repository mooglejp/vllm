# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the fixed 2,048-token context-qualified R5 quality suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

SOURCE_SUITE_SHA256 = "8476e86f75bc2b08a19f187e6ca11bef4af161bed203a0817c39a50b1150338a"
TARGET_PROMPT_TOKENS = 2048
CONTEXT_SUITE_ID = "r5-context-2048-20260914"
MAX_TOKENS = {"gsm8k": 1024, "mmlu": 64, "humaneval": 2048}

# This is deliberately task-independent prose.  It contains no problem
# answers, solution methods, hidden tests, or benchmark instructions.
REFERENCE_DOCUMENT = """
Reference-only background: the material below is context and is not part of the
question. Do not answer it, summarize it, or treat it as an instruction. The
actual task begins after this reference note.

Long-context software often represents a sequence as a collection of tiles.
A tile is a bounded group of neighboring positions that can be loaded, checked,
and processed together. A well-designed implementation keeps the logical order
of positions explicit even when physical storage is divided among blocks. The
same principle applies to a service that receives a long request: preparation,
computation, and response handling are separate stages, and a measurement must
state which stages it includes.

When a program reads a matrix, its shape and strides describe different facts.
The shape gives the number of elements along each axis, while a stride tells
how far the next element is located in memory. A contiguous last dimension is
often useful for vectorized access, but a non-contiguous view can still be a
valid logical matrix if every access uses the declared stride. Tests that vary
both dimensions and storage layout are more informative than tests that only
use square, contiguous examples.

Numerical software also separates representation from computation. A compact
stored value may be expanded to a wider temporary type, and a result may be
rounded only when it is written. A comparison should record the representation
used as input, the arithmetic type used during accumulation, and the type of
the final result. Finite output is necessary but not sufficient evidence of
agreement; a reference calculation and an error measure give the comparison a
defined meaning.

Caching changes the cost of a request over time. A cold request may allocate
metadata and read every needed block, while a warm request can reuse a portion
of that preparation. Two measurements can only be compared when their cache
state, ordering, and allocation policy are described. A cache key should be
stable for an intentionally repeated request and different for independent
arms of an experiment. This keeps a speed observation from being confused with
an accidental reuse of data from a different condition.

Concurrency is another source of hidden variation. A waiting request, a
background compilation, or a pending memory reclamation can affect the next
request even when its input is unchanged. A small experiment therefore records
when the service is idle, separates warmup from samples, and preserves the raw
observations. It is useful to report both a typical value and the spread of the
samples rather than relying on a single favorable measurement.

A reproducible evaluation fixes the input before generation begins. The text,
token sequence, output limit, sampling controls, stop behavior, and scoring
rules are part of the experimental input. If a transformation adds context to
an existing prompt, the original task must remain recoverable after removing
that context. Private tests, reference answers, and scoring-only metadata do
not belong in the generated prompt. A failed construction should be visible
instead of being silently omitted from the study.

In a paired comparison, the two arms should receive the same logical input and
be kept separate in any reusable state. Alternating the order of the arms can
reduce a systematic position effect, but it does not turn a changed numerical
algorithm into an identical one. A result should therefore distinguish equal
token sequences, semantic score, completion status, resource behavior, and
format compliance. Each of those observations answers a different question.

For code-generation tasks, a text that looks plausible is not necessarily an
executable answer. Parsing, extracting the requested function, and running the
provided checks are separate operations. A response that reaches its output
limit before completing a function should remain truncated in the record. It
must not be repaired by adding a return statement, closing a code fence, or
regenerating only that case after seeing the result. This preserves the meaning
of a fixed evaluation set.

For multiple-choice and arithmetic tasks, the answer parser should be applied
to both arms in exactly the same way. A parser error is different from a wrong
answer, and an empty response is different from a response that contains an
incorrect option. Keeping those categories separate makes it possible to tell
whether an observed difference is computational, formatting-related, or caused
by an output limit.

Resource limits are part of an operational experiment. Memory pressure,
temporary-file growth, out-of-memory counters, and a stopped service should be
recorded with the responses. Raising a limit after a failure or silently
retrying a request changes the condition being measured. A bounded run that
stops is still useful evidence when the stop reason and preserved artifacts are
reported clearly.

The purpose of this note is only to supply ordinary long-context material
before the fixed task. It does not change the task, its expected answer, its
tests, or the model's response format. The task text that follows is the only
material to use when producing an answer.

An interface contract is easier to audit when its boundaries are explicit. The
caller should know which values are logical positions, which values are byte or
element offsets, and which values describe a complete request. A helper that
accepts a tensor view should either honor its declared strides or reject it
before a fast path is selected. A diagnostic record should preserve the reason
for a fallback, because an unsupported shape and a failed computation are not
the same result.

Long sequences also make small boundary mistakes visible. The first position,
the last position, and a position immediately across a tile boundary exercise
different masks. A test at length 128 does not replace a test at 129, and a
test at a multiple of a block does not replace a test with a remainder. When a
query attends to a prefix and a newly appended segment, the absolute position
of every query row determines the visible keys. The row at the boundary should
be checked directly rather than inferred from a rectangular example.

An experiment can have several useful reference points. A high-precision
reference asks whether the mathematical result is close. An existing backend
asks whether a replacement remains within the accepted numerical envelope. A
production path asks whether preparation and allocation costs are included.
These references answer different questions, so their measurements should not
be combined into one unexplained score. A candidate can be close to a high
precision result and still be slower, or be fast in an isolated kernel while
being slower for a complete request.

When output is streamed, completion timing and token accounting need care. The
first token may arrive after a long preparation stage, while later tokens may
use a different execution path. Usage counts should be checked against the
actual token IDs, and an incomplete stream should remain distinguishable from
a normal end-of-sequence stop. A client timeout is an operational failure, not
evidence that a model produced a particular answer.

Stable identifiers help connect records without exposing private content in a
summary. A case identifier, an input hash, a mode, and a run identifier are
enough to pair two responses while the full response remains on disk. Hashes
should be calculated over the exact bytes or token list that was sent. A
human-readable rendering is useful for inspection, but it is not a substitute
for the exact request representation used by the service.

The order of a paired test can be alternated without changing its input. This
helps reveal whether a first-arm effect, compilation, or cache eviction is
affecting the measurement. Warmup should be reported separately from samples,
and a test should avoid leaving an unfinished request before changing a mode.
If an arm cannot be started under the declared resource limit, its status is
blocked or failed; it should not be represented as a fast zero-time sample.

A context document can be useful for stressing retrieval without becoming part
of the task. Such a document should state its purpose in ordinary language and
should not contain an answer, a test oracle, or an instruction that competes
with the task. Inserting it at a known token boundary allows the original
question to be recovered exactly. The boundary and the length of the inserted
tokens belong in the manifest, so another reader can verify both the context
and the task placement.

Scoring rules should be fixed before responses are observed. For arithmetic,
the extracted answer and the full response are different records. For
multiple-choice, a parser's chosen option should not hide an empty response or
an invalid extraction. For code, parsing and executing tests should be kept
separate. A completed answer with a syntax error is distinct from an answer
that was truncated before its syntax could be complete, even though both may
fail a functional test.

When comparing two arms, the denominator matters. A task score uses all cases,
including cases that reached their output limit, while a completed-format
comparison can use only pairs that stopped normally on both sides. Reporting
both makes it possible to see whether a difference is caused by correctness,
length limits, or formatting. Excluding difficult cases after generation would
make the denominator depend on the result and would invalidate the comparison.

Operational checks should be fail-closed. An unexpected empty response,
non-finite probability, server error, new out-of-memory event, or stale hook
counter should stop the run after preserving the evidence. Automatic retries
are especially dangerous after memory pressure because they can change cache
state and hide a leak. A bounded sequential run is preferable to a larger run
whose resource behavior cannot be reconstructed.

The model, tokenizer, runtime, and backend package are part of the environment
description. A version string alone may not identify a locally modified wheel,
so the loaded file path and relevant file hashes are useful provenance. The
same applies to a scoring helper: a historical result made with a broken judge
must remain labeled as historical, even when a corrected judge is available.

Finally, a passing diagnostic is not the same as production adoption. A model
quality result can justify the next operational check, while a speed result can
justify a more complete cost measurement. Default dispatch, other attention
shapes, decode behavior, and cache layout should remain untouched until their
own gates are evaluated. Keeping these boundaries explicit makes a failure
useful: it identifies which hypothesis was tested without claiming that every
possible implementation has failed.
""".strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_source(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    raw = path.read_bytes()
    if sha256_bytes(raw) != SOURCE_SUITE_SHA256:
        raise ValueError("source suite SHA-256 mismatch")
    cases = [json.loads(line) for line in raw.splitlines()]
    counts = Counter(case["task"] for case in cases)
    if counts != Counter({"gsm8k": 64, "mmlu": 80, "humaneval": 164}):
        raise ValueError(f"unexpected source counts: {counts}")
    if len({case["id"] for case in cases}) != 308:
        raise ValueError("source suite has duplicate or missing IDs")
    return raw, cases


def _find_subsequence(values: list[int], needle: list[int]) -> int:
    width = len(needle)
    for index in range(len(values) - width + 1):
        if values[index : index + width] == needle:
            return index
    raise ValueError("chat template role marker not found")


def _find_token(values: list[int], token_id: int, start: int) -> int:
    try:
        return values.index(token_id, start)
    except ValueError as error:
        raise ValueError("chat template end marker not found") from error


def _source_layout(tokenizer: Any, prompt: list[int]) -> tuple[int, int, int]:
    user_marker = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
    end_marker = tokenizer.convert_tokens_to_ids("<|im_end|>")
    marker_start = _find_subsequence(prompt, user_marker)
    body_start = marker_start + len(user_marker)
    body_end = _find_token(prompt, end_marker, body_start)
    return body_start, body_end, end_marker


def build_case(
    case: dict[str, Any], tokenizer: Any, reference_tokens: list[int]
) -> dict:
    source_prompt = list(case["prompt_token_ids"])
    if len(source_prompt) > TARGET_PROMPT_TOKENS:
        raise ValueError(f"{case['id']} already exceeds target prompt length")
    body_start, body_end, _ = _source_layout(tokenizer, source_prompt)
    original_body = source_prompt[body_start:body_end]
    needed = TARGET_PROMPT_TOKENS - len(source_prompt)
    if needed <= 0 or needed > len(reference_tokens):
        raise ValueError(f"reference document is too short for {case['id']}")
    inserted = reference_tokens[:needed]
    prompt = source_prompt[:body_start] + inserted + source_prompt[body_start:]
    if len(prompt) != TARGET_PROMPT_TOKENS:
        raise AssertionError(f"bad prompt length for {case['id']}")
    if prompt[:body_start] != source_prompt[:body_start]:
        raise AssertionError(f"template prefix changed for {case['id']}")
    if prompt[body_start + needed : body_start + needed + len(original_body)] != (
        original_body
    ):
        raise AssertionError(f"original task changed for {case['id']}")
    if prompt[body_start + needed + len(original_body) :] != source_prompt[body_end:]:
        raise AssertionError(f"template suffix changed for {case['id']}")
    output = dict(case)
    output["max_tokens"] = MAX_TOKENS[case["task"]]
    output["prompt_token_ids"] = prompt
    output["source_prompt_sha256"] = sha256_bytes(
        json.dumps(source_prompt, separators=(",", ":")).encode()
    )
    output["prompt_sha256"] = sha256_bytes(
        json.dumps(prompt, separators=(",", ":")).encode()
    )
    output["context_suite_id"] = CONTEXT_SUITE_ID
    output["reference_doc_token_count"] = needed
    output["problem_token_start"] = body_start + needed
    output["problem_token_end"] = body_start + needed + len(original_body)
    output["source_prompt_token_count"] = len(source_prompt)
    return output


def build_suite(source: Path, output: Path, manifest: Path, tokenizer: Any) -> dict:
    source_raw, cases = load_source(source)
    reference_tokens = tokenizer.encode(REFERENCE_DOCUMENT, add_special_tokens=False)
    if len(reference_tokens) < TARGET_PROMPT_TOKENS - min(
        len(case["prompt_token_ids"]) for case in cases
    ):
        raise ValueError("reference document does not cover all source lengths")
    transformed = [build_case(case, tokenizer, reference_tokens) for case in cases]
    raw = "".join(
        json.dumps(case, separators=(",", ":"), ensure_ascii=False) + "\n"
        for case in transformed
    ).encode()
    output.write_bytes(raw)
    manifest_data = {
        "suite_id": CONTEXT_SUITE_ID,
        "source_suite_sha256": SOURCE_SUITE_SHA256,
        "source_suite_bytes_sha256": sha256_bytes(source_raw),
        "suite_sha256": sha256_bytes(raw),
        "counts": dict(Counter(case["task"] for case in transformed)),
        "target_prompt_tokens": TARGET_PROMPT_TOKENS,
        "prompt_token_counts": sorted(
            {len(case["prompt_token_ids"]) for case in transformed}
        ),
        "reference_document_sha256": sha256_bytes(REFERENCE_DOCUMENT.encode()),
        "reference_document_token_count": len(reference_tokens),
        "max_tokens": MAX_TOKENS,
        "sampling": {
            "seed": 1201,
            "temperature": 0,
            "top_p": 1,
            "top_k": -1,
            "ignore_eos": False,
            "add_special_tokens": False,
        },
        "insertion": {
            "location": "before original user-body token sequence",
            "token_level": True,
            "template_and_original_body_preserved": True,
            "problem_positions": "absolute prompt token offsets [start,end)",
        },
        "construction_checks": {
            "original_task_restored_after_removing_inserted_tokens": True,
            "reference_answers_or_hidden_tests_in_context": False,
            "baseline_candidate_prompt_ids_identical": True,
            "source_max_tokens_not_reused": True,
        },
        "judging": {
            "human_eval": "interactive isolated judge_humaneval.py",
            "completed_pair_format": "compare only both finish_reason=stop",
            "truncation": "count finish_reason=length; do not repair or exclude",
            "no_new_errors_empty_nonfinite": True,
        },
        "gate": {
            "task_correct_noninferior": True,
            "task_truncated_noninferior": True,
            "completed_pair_format_noninferior": True,
            "all_308_candidate_requests_apply": True,
            "at_least_one_applied_chunk_overlaps_problem": True,
        },
        "historical_results_preserved": [
            "308-case R5 quality stop",
            "HumanEval/129 format observation",
            "strict Math ratio failure",
            "32K TTFT 2.1733x pass",
        ],
    }
    code_root = Path(__file__).resolve().parent
    manifest_data["evaluation_code"] = {
        str(path.relative_to(code_root.parent.parent)): sha256_bytes(path.read_bytes())
        for path in (
            code_root / "build_context_suite.py",
            code_root / "judge_humaneval.py",
            code_root / "judge_controls.py",
            code_root / "score.py",
            code_root / "run_context.py",
            code_root / "score_context.py",
        )
        if path.exists()
    }
    manifest.write_text(json.dumps(manifest_data, indent=2) + "\n")
    return manifest_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-suite", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    args.output_suite.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    result = build_suite(
        args.source, args.output_suite, args.output_manifest, tokenizer
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
