# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the fixed 32K R5 content-retention diagnostic suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

TARGET_PROMPT_TOKENS = 32768
MAX_TOKENS = 128
SEED = 1201
SUITE_ID = "r5-retention-32k-20260915"

CONTENT_SPECS = (
    {
        "content_id": "archive_lantern",
        "theme": "lantern archive",
        "identifier": "ARCHIVE-LANTERN-7Q4",
        "answer": "R7N4-K2P9-V6X1",
    },
    {
        "content_id": "moss_circuit",
        "theme": "moss circuit",
        "identifier": "MOSS-CIRCUIT-3H8",
        "answer": "Q5W8-L1D6-Z9C2",
    },
    {
        "content_id": "orbit_kestrel",
        "theme": "orbit kestrel",
        "identifier": "ORBIT-KESTREL-9M2",
        "answer": "T4B7-Y8F3-N6J0",
    },
)

POSITIONS = ("early", "middle", "late")
POSITION_RATIOS = {"early": 0.10, "middle": 0.50, "late": 0.90}

# These are ordinary, answer-free sentences.  Section numbers make each
# selected sentence inspectable without introducing a synthetic symbol stream.
FILLER_SENTENCES = (
    (
        "The register describes how a careful reviewer keeps neighboring records "
        "in logical order."
    ),
    (
        "A measured transfer is easier to audit when its source, destination, and "
        "boundary are recorded."
    ),
    (
        "The note separates preparation from computation so that a timing result "
        "has a clear scope."
    ),
    (
        "A stable label helps connect a stored observation to the exact request "
        "that produced it."
    ),
    (
        "The archive favors complete sentences because a boundary should not split "
        "a meaningful statement."
    ),
    (
        "A long document can remain readable when each section states its subject "
        "before giving its detail."
    ),
    (
        "The review process records both the ordinary path and the reason a fallback "
        "was selected."
    ),
    (
        "A cache observation is useful only when the ordering and reuse policy are "
        "stated beside it."
    ),
    (
        "The ledger treats a representation and the computation performed on it as "
        "separate facts."
    ),
    (
        "A boundary check includes the first position, the last position, and the "
        "position after a tile."
    ),
    (
        "The operator compares paired inputs before drawing a conclusion about a "
        "changed execution path."
    ),
    (
        "A resource report keeps memory pressure distinct from a numerical or "
        "semantic result."
    ),
    (
        "The document uses a fixed vocabulary so that the retrieval question has one "
        "unambiguous target."
    ),
    (
        "A sequential observation avoids attributing a pending request to the next "
        "independent sample."
    ),
    (
        "The archive records an interval explicitly rather than inferring it from a "
        "rounded percentage."
    ),
    (
        "A reproducible fixture fixes its text and token sequence before any response "
        "is observed."
    ),
    (
        "The review can compare a complete response with a length-limited response "
        "without repairing either."
    ),
    (
        "A compact summary points back to the full record instead of replacing the "
        "evidence it describes."
    ),
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _template_parts(tokenizer: Any) -> tuple[list[int], list[int]]:
    prefix = _encode(tokenizer, "<|im_start|>user\n")
    suffix = _encode(
        tokenizer,
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if list(rendered) != prefix + suffix:
        raise ValueError("non-thinking chat template does not match fixed parts")
    return prefix, suffix


def _sentence_bank(tokenizer: Any, spec: dict[str, str]) -> list[tuple[str, list[int]]]:
    bank: list[tuple[str, list[int]]] = []
    for ordinal in range(1024):
        sentence = (
            f"Section {ordinal:04d} of the {spec['theme']} record: "
            f"{FILLER_SENTENCES[ordinal % len(FILLER_SENTENCES)]}\n"
        )
        tokens = _encode(tokenizer, sentence)
        if tokens:
            bank.append((sentence, tokens))
    if len({len(tokens) for _, tokens in bank}) < 4:
        raise ValueError("filler sentence bank has insufficient token lengths")
    return bank


def _exact_sentence_tokens(
    target: int,
    bank: list[tuple[str, list[int]]],
    start_ordinal: int,
) -> tuple[str, list[int]]:
    """Return complete filler sentences with exactly ``target`` tokens."""

    if target < 0:
        raise ValueError(f"negative filler token target: {target}")
    if target == 0:
        return "", []
    by_length: dict[int, list[int]] = {}
    for index, (_, tokens) in enumerate(bank):
        by_length.setdefault(len(tokens), []).append(index)
    lengths = sorted(by_length)
    previous: list[tuple[int, int] | None] = [None] * (target + 1)
    previous[0] = (-1, -1)
    for total in range(target + 1):
        if previous[total] is None:
            continue
        for length in lengths:
            next_total = total + length
            if next_total > target or previous[next_total] is not None:
                continue
            previous[next_total] = (total, length)
            if next_total == target:
                break
        if previous[target] is not None:
            break
    if previous[target] is None:
        raise ValueError(f"could not fill exactly {target} tokens")

    chosen_lengths: list[int] = []
    total = target
    while total:
        prior, length = previous[total]  # type: ignore[misc]
        chosen_lengths.append(length)
        total = prior
    chosen_lengths.reverse()

    pieces: list[str] = []
    tokens: list[int] = []
    counters: dict[int, int] = {}
    for length in chosen_lengths:
        candidates = by_length[length]
        offset = counters.get(length, 0)
        index = candidates[(start_ordinal + offset) % len(candidates)]
        counters[length] = offset + 1
        sentence, sentence_tokens = bank[index]
        pieces.append(sentence)
        tokens.extend(sentence_tokens)
    if len(tokens) != target:
        raise AssertionError("filler token count changed")
    return "".join(pieces), tokens


def _find_subsequence(values: list[int], needle: list[int], label: str) -> int:
    if not needle:
        raise ValueError(f"empty token sequence for {label}")
    matches = [
        index
        for index in range(len(values) - len(needle) + 1)
        if values[index : index + len(needle)] == needle
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {label} token interval, found {len(matches)}")
    return matches[0]


def _build_case(
    tokenizer: Any,
    prefix: list[int],
    suffix: list[int],
    spec: dict[str, str],
    position: str,
) -> dict[str, Any]:
    bank = _sentence_bank(tokenizer, spec)
    answer = spec["answer"]
    identifier = spec["identifier"]
    target_sentence = (
        f'Registry entry: identifier "{identifier}" has registration number '
        f'"{answer}".\n'
    )
    question = (
        f"Question: The document contains one registry entry for identifier "
        f"{identifier}. Return only its registration number, on one line, with "
        "no explanation."
    )
    target_tokens = _encode(tokenizer, target_sentence)
    question_tokens = _encode(tokenizer, question)
    body_target = TARGET_PROMPT_TOKENS - len(prefix) - len(suffix)
    document_target = body_target - len(question_tokens)
    prefix_target = (
        round(document_target * POSITION_RATIOS[position]) - len(target_tokens) // 2
    )
    prefix_target = max(0, prefix_target)
    suffix_target = document_target - prefix_target - len(target_tokens)
    if suffix_target < 0:
        raise ValueError(f"document target too short for {spec['content_id']}")
    prefix_text, prefix_tokens = _exact_sentence_tokens(prefix_target, bank, 0)
    suffix_text, suffix_tokens = _exact_sentence_tokens(suffix_target, bank, 4096)
    body_tokens = prefix_tokens + target_tokens + suffix_tokens + question_tokens
    prompt_tokens = prefix + body_tokens + suffix
    if len(prompt_tokens) != TARGET_PROMPT_TOKENS:
        raise AssertionError(f"wrong prompt length for {spec['content_id']}/{position}")
    body_text = tokenizer.decode(body_tokens, clean_up_tokenization_spaces=False)
    if _encode(tokenizer, body_text) != body_tokens:
        raise ValueError(
            f"body token round-trip failed for {spec['content_id']}/{position}"
        )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": body_text}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if list(rendered) != prompt_tokens:
        raise ValueError(
            f"chat template round-trip failed for {spec['content_id']}/{position}"
        )
    if body_text.count(target_sentence.strip()) != 1:
        raise ValueError("target sentence is not unique")
    if body_text.count(answer) != 1:
        raise ValueError("answer is not unique in document")
    if question.count(answer):
        raise ValueError("answer leaked into question")
    if body_text.count(identifier) != 2:
        raise ValueError("identifier must occur once in document and once in question")
    for other in CONTENT_SPECS:
        if other["answer"] != answer and body_text.count(other["answer"]):
            raise ValueError("another content answer leaked into filler")

    target_offset = _find_subsequence(body_tokens, target_tokens, "target")
    question_offset = _find_subsequence(body_tokens, question_tokens, "question")
    body_start = len(prefix)
    document_start = body_start
    document_end = body_start + question_offset
    target_start = body_start + target_offset
    question_start = body_start + question_offset
    return {
        "case_id": f"{spec['content_id']}-{position}",
        "content_id": spec["content_id"],
        "position": position,
        "position_ratio_target": POSITION_RATIOS[position],
        "position_ratio_actual": (target_start - document_start)
        / (document_end - document_start),
        "identifier": identifier,
        "answer": answer,
        "target_sentence": target_sentence.strip(),
        "prompt_token_ids": prompt_tokens,
        "prompt_tokens": len(prompt_tokens),
        "max_tokens": MAX_TOKENS,
        "document_token_start": document_start,
        "document_token_end": document_end,
        "target_token_start": target_start,
        "target_token_end": target_start + len(target_tokens),
        "question_token_start": question_start,
        "question_token_end": question_start + len(question_tokens),
        "document_token_count": question_offset,
        "prompt_sha256": sha256_bytes(
            json.dumps(prompt_tokens, separators=(",", ":")).encode()
        ),
        "body_sha256": sha256_bytes(body_text.encode()),
        "target_token_count": len(target_tokens),
        "question_token_count": len(question_tokens),
        "prefix_text_sha256": sha256_bytes(prefix_text.encode()),
        "suffix_text_sha256": sha256_bytes(suffix_text.encode()),
    }


def build_suite(
    tokenizer: Any, model_path: Path, output_suite: Path, output_manifest: Path
) -> dict[str, Any]:
    prefix, suffix = _template_parts(tokenizer)
    cases = [
        _build_case(tokenizer, prefix, suffix, spec, position)
        for spec in CONTENT_SPECS
        for position in POSITIONS
    ]
    raw = "".join(
        json.dumps(case, separators=(",", ":"), ensure_ascii=False) + "\n"
        for case in cases
    ).encode()
    output_suite.write_bytes(raw)
    tokenizer_files = {}
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
    ):
        path = model_path / name
        if path.exists():
            tokenizer_files[name] = sha256_path(path)
    manifest = {
        "suite_id": SUITE_ID,
        "suite_sha256": sha256_bytes(raw),
        "case_count": len(cases),
        "content_count": len(CONTENT_SPECS),
        "positions": list(POSITIONS),
        "target_prompt_tokens": TARGET_PROMPT_TOKENS,
        "max_tokens": MAX_TOKENS,
        "sampling": {
            "seed": SEED,
            "temperature": 0,
            "top_p": 1,
            "top_k": -1,
            "ignore_eos": False,
            "add_special_tokens": False,
            "enable_thinking": False,
        },
        "model_path": str(model_path),
        "tokenizer_files": tokenizer_files,
        "template": {
            "chat_template": "model tokenizer config",
            "add_generation_prompt": True,
            "enable_thinking": False,
            "document_and_question_in_one_user_message": True,
        },
        "construction": {
            "document_target_positions": POSITION_RATIOS,
            "target_sentence_occurs_once_in_document": True,
            "answer_occurs_once_in_document": True,
            "answer_not_in_question_or_filler": True,
            "no_contradictory_content_answers": True,
            "prompt_ids_equal_for_baseline_candidate": True,
            "whole_prompt_includes_template_document_question": True,
            "token_boundaries_do_not_split_sentences": True,
        },
        "scoring": {
            "main": "response_text.strip() == answer exactly",
            "normalization": "remove only leading/trailing whitespace",
            "contains_answer": "auxiliary only; never substitutes for exact match",
            "retry_or_repair": False,
        },
        "execution": {
            "arms": ["baseline", "candidate"],
            "order": "B/C then C/B alternating by case",
            "requests_per_case_per_arm": 1,
            "cache_salt": "unique per suite/run/mode/case; no A/B sharing",
            "ram_limit_bytes": 16 * 2**30,
            "swap_limit_bytes": 0,
            "candidate_scope": "existing target continuation attention only",
        },
        "historical_results_preserved": {
            "additional_quality_gate": "failed and unchanged",
            "cold_32k_ttft_speedup": "2.1733x and unchanged",
        },
    }
    code_root = Path(__file__).resolve().parent
    manifest["evaluation_code"] = {
        str(path.relative_to(code_root.parent.parent)): sha256_path(path)
        for path in (
            code_root / "build_retention_suite.py",
            code_root / "run_retention.py",
            code_root / "score_retention.py",
            code_root / "retention_environment.py",
            code_root.parent.parent / "benchmarks/r5_model_ab/sitecustomize.py",
            code_root.parent.parent
            / "benchmarks/benchmark_gfx1201_r5_model_ab_hook.py",
        )
        if path.exists()
    }
    output_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-suite", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    args.output_suite.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            build_suite(tokenizer, args.model, args.output_suite, args.output_manifest),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
