# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture runtime and source provenance for the 32K retention run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_value(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def capture(
    output: Path,
    model: Path,
    source_root: Path,
    source_git_head: str | None,
    source_git_status: str | None,
) -> dict:
    import flash_attn
    import torch

    files = {}
    for path in (
        source_root / "benchmarks/r5_quality/build_retention_suite.py",
        source_root / "benchmarks/r5_quality/run_retention.py",
        source_root / "benchmarks/r5_quality/score_retention.py",
        source_root / "benchmarks/r5_model_ab/sitecustomize.py",
        source_root / "benchmarks/benchmark_gfx1201_r5_model_ab_hook.py",
        model / "tokenizer.json",
        model / "tokenizer_config.json",
        model / "chat_template.jinja",
        model / "config.json",
    ):
        if path.exists():
            files[str(path)] = sha256(path)
    root = Path("/sys/fs/cgroup")
    data = {
        "source_root": str(source_root),
        "git_head": source_git_head or git_value(source_root, "rev-parse", "HEAD"),
        "git_status_porcelain": source_git_status
        if source_git_status is not None
        else git_value(source_root, "status", "--short"),
        "model_path": str(model),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "flash_attn": importlib.metadata.version("flash-attn"),
        "flash_file": str(Path(flash_attn.__file__).resolve()),
        "flash_forward_file": str(
            Path(flash_attn.flash_attn_varlen_func.__code__.co_filename).resolve()
        ),
        "files_sha256": files,
        "limits": {
            name: (root / name).read_text().strip()
            for name in ("memory.max", "memory.swap.max")
        },
        "settings": {
            "tp": 1,
            "kv": "turboquant_k8v4",
            "mtp_tokens": 2,
            "adaptive_verification": False,
            "fused_mxfp4_decode": True,
            "compilation_mode": 0,
            "graph": "FULL_DECODE_ONLY",
            "chunk_budget": 256,
            "max_num_seqs": 1,
            "prompt_tokens": 32768,
            "max_tokens": 128,
            "temperature": 0,
            "seed": 1201,
            "enable_thinking": False,
            "profiler_started": False,
            "production_changes": False,
        },
    }
    output.write_text(json.dumps(data, indent=2) + "\n")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--git-head")
    parser.add_argument("--git-status")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            capture(
                args.output,
                args.model,
                args.source_root,
                args.git_head,
                args.git_status,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
