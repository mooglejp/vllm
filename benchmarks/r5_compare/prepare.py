# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare a cross-runtime vLLM/llama.cpp comparison suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

SPEED_TARGETS = (128, 1024, 4096, 16384, 32768)
SPEED_BLOCKS = (
    """
    監視システムの運用記録では、観測、判定、保存、再確認の境界を明確に
    する。入力の順序と時刻を保持し、推測した値は測定値と混ぜない。障害が
    起きた場合は、通常の誤判定と、サービスが応答できなかった状態を別々に
    記録する。
    """,
    """
    def summarize_window(values: list[float], width: int) -> dict[str, float]:
        window = values[-width:] if width else []
        if not window:
            return {"count": 0.0, "mean": 0.0, "last": 0.0}
        return {
            "count": float(len(window)),
            "mean": sum(window) / len(window),
            "last": window[-1],
        }
    """,
    """
    A paged attention implementation can store a logical sequence in physical
    blocks. The block table, the number of visible keys, and the query's
    absolute position are independent pieces of metadata. A benchmark should
    state whether it measures only the arithmetic kernel or also includes
    cache reads, metadata construction, synchronization, and response delivery.
    """,
    """
    When comparing two serving systems, a first request can include compilation,
    allocator growth, model page faults, and prompt-cache construction. Those
    effects are useful operational observations, but they are not interchangeable
    with a steady-state sample. Keep the first request separate and retain every
    raw timestamp so that the reported median can be reconstructed.
    """,
    """
    エッジ側のログは日本語、英語、短いコード片が混在することがある。
    単語の頻度だけでなく、節の境界、箇条書き、識別子、数値、改行を保った
    入力で評価すると、実際の対話に近い前処理と長文保持の負荷を観測できる。
    """,
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def extract_user(tokenizer: Any, token_ids: list[int], case_id: str) -> str:
    rendered = tokenizer.decode(token_ids, skip_special_tokens=False)
    marker = "<|im_start|>user\n"
    end_marker = "<|im_end|>"
    starts = [
        index for index in range(len(rendered)) if rendered.startswith(marker, index)
    ]
    if len(starts) != 1:
        raise ValueError(f"expected one user message for {case_id}, got {len(starts)}")
    body_start = starts[0] + len(marker)
    body_end = rendered.find(end_marker, body_start)
    if body_end < 0:
        raise ValueError(f"missing user end marker for {case_id}")
    body = rendered[body_start:body_end]
    if "<|im_start|>" in body or "<|im_end|>" in body:
        raise ValueError(f"nested chat marker in user body for {case_id}")
    return body


def _messages(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


def make_speed_prompt(tokenizer: Any, target: int) -> tuple[str, int]:
    header = (
        "Read the following mixed-language engineering notebook. Continue with "
        "a detailed but coherent technical analysis of the material. Preserve "
        "identifiers and numerical relationships; do not stop after one sentence.\n\n"
    )
    parts = [header]
    index = 0
    while len(tokenizer.encode("".join(parts), add_special_tokens=False)) < target:
        block = SPEED_BLOCKS[index % len(SPEED_BLOCKS)].strip()
        parts.append(f"\nSection {index + 1}:\n{block}\n")
        index += 1
    text = "".join(parts)
    return text, len(tokenizer.encode(text, add_special_tokens=False))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def build_suite(
    output: Path,
    tokenizer: Any,
    context_path: Path,
    retention_path: Path,
    model_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    context_rows = load_jsonl(context_path)
    retention_rows = load_jsonl(retention_path)
    quality = []
    for row in context_rows:
        text = extract_user(tokenizer, row["prompt_token_ids"], row["id"])
        quality.append(
            {
                "id": row["id"],
                "task": row["task"],
                "max_tokens": row["max_tokens"],
                "reference": row["reference"],
                "prompt_text": text,
                "messages": _messages(text),
                "source_prompt_sha256": row["source_prompt_sha256"],
                "vllm_prompt_sha256": row["prompt_sha256"],
            }
        )
    retention = []
    for row in retention_rows:
        text = extract_user(tokenizer, row["prompt_token_ids"], row["case_id"])
        retention.append(
            {
                "id": row["case_id"],
                "content_id": row["content_id"],
                "position": row["position"],
                "identifier": row["identifier"],
                "answer": row["answer"],
                "max_tokens": row["max_tokens"],
                "prompt_text": text,
                "messages": _messages(text),
                "vllm_prompt_sha256": row["prompt_sha256"],
            }
        )
    speed = []
    for target in SPEED_TARGETS:
        text, raw_tokens = make_speed_prompt(tokenizer, target)
        speed.append(
            {
                "id": f"speed-{target}",
                "target_prompt_tokens": target,
                "vllm_raw_content_tokens": raw_tokens,
                "max_tokens": 256,
                "prompt_text": text,
                "messages": _messages(text),
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("speed", speed),
        ("quality", quality),
        ("retention", retention),
    ):
        path = output / f"{name}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
    manifest = {
        "comparison_id": "r5-vllm-vs-production-llama-20260914",
        "vllm_commit": _git_head(repo_root),
        "model_path": str(model_path),
        "tokenizer": {
            "tokenizer_json_sha256": sha256_file(model_path / "tokenizer.json"),
            "tokenizer_config_sha256": sha256_file(
                model_path / "tokenizer_config.json"
            ),
            "chat_template_sha256": sha256_file(model_path / "chat_template.jinja"),
        },
        "source_suites": {
            "context": {"path": str(context_path), "sha256": sha256_file(context_path)},
            "retention": {
                "path": str(retention_path),
                "sha256": sha256_file(retention_path),
            },
        },
        "generated_suites": {
            name: {
                "path": str(output / f"{name}.jsonl"),
                "sha256": sha256_file(output / f"{name}.jsonl"),
                "cases": len(rows),
            }
            for name, rows in (
                ("speed", speed),
                ("quality", quality),
                ("retention", retention),
            )
        },
        "speed": {
            "targets": SPEED_TARGETS,
            "samples_per_target": 5,
            "max_tokens": 256,
            "ignore_eos": True,
            "prefix_reuse": False,
            "warmup_excluded": True,
        },
        "quality": {
            "counts": {"gsm8k": 64, "mmlu": 80, "humaneval": 164},
            "max_tokens": {"gsm8k": 1024, "mmlu": 64, "humaneval": 2048},
            "thinking": (
                "vLLM non-thinking request; llama.cpp production reasoning setting "
                "retained and recorded"
            ),
            "sampling": {"temperature": 0, "seed": 1201, "normal_eos": True},
        },
        "retention": {
            "cases": 9,
            "positions": ["early", "middle", "late"],
            "max_tokens": 128,
            "exact_match": "strip leading/trailing whitespace only",
        },
        "comparison_limits": {
            "same_model": False,
            "same_quantization": False,
            "production_adoption": False,
            "kernel_changes": False,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--retention", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    result = build_suite(
        args.output,
        tokenizer,
        args.context,
        args.retention,
        args.model,
        args.repo_root,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
