# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture immutable runtime facts for the vLLM/llama.cpp comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def command(args: list[str], timeout: int = 30) -> str:
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"unavailable: {error}"
    output = result.stdout.strip()
    if result.returncode != 0:
        error = result.stderr.strip()
        return f"exit={result.returncode}: {error or output}"
    return output


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, hash_file: bool = False) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return record
    stat = path.stat()
    record["size_bytes"] = stat.st_size
    if hash_file:
        record["sha256"] = sha256_file(path)
    return record


def docker_json(container: str, template: str) -> Any:
    raw = command(["docker", "inspect", "--format", template, container])
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--vllm-model", type=Path, required=True)
    parser.add_argument("--llama-container", default="ai-llama")
    parser.add_argument(
        "--llama-model",
        type=Path,
        default=Path(
            "/srv/ai/models/llm/gguf/huihui-qwen38-ud/"
            "Huihui-Qwen3.8-27B-abliterated-UD-Q4_K_XL.gguf"
        ),
    )
    parser.add_argument(
        "--llama-mmproj",
        type=Path,
        default=Path("/srv/ai/models/llm/gguf/huihui-qwen38-ud/mmproj-model-bf16.gguf"),
    )
    parser.add_argument(
        "--llama-draft",
        type=Path,
        default=Path(
            "/srv/ai/models/llm/gguf/huihui-qwen38-ud/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"
        ),
    )
    args = parser.parse_args()

    inspect_config = docker_json(args.llama_container, "{{json .Config}}")
    inspect_host = docker_json(args.llama_container, "{{json .HostConfig}}")
    inspect_state = docker_json(args.llama_container, "{{json .State}}")
    image = inspect_config.get("Image") if isinstance(inspect_config, dict) else None
    image_config = docker_json(image, "{{json .RepoDigests}}") if image else None
    image_labels = docker_json(image, "{{json .Config.Labels}}") if image else None
    safe_config = {}
    if isinstance(inspect_config, dict):
        safe_config = {
            key: inspect_config.get(key)
            for key in ("User", "Entrypoint", "Cmd", "WorkingDir", "Image")
        }
        safe_config["Env_names"] = sorted(
            item.split("=", 1)[0] for item in inspect_config.get("Env", [])
        )
    safe_host = {}
    if isinstance(inspect_host, dict):
        safe_host = {
            key: inspect_host.get(key)
            for key in (
                "Binds",
                "Devices",
                "GroupAdd",
                "NetworkMode",
                "PortBindings",
                "ReadonlyRootfs",
                "SecurityOpt",
                "ShmSize",
                "Tmpfs",
                "Memory",
                "MemorySwap",
            )
        }
    repo = args.repo_root
    result: dict[str, Any] = {
        "comparison_id": "r5-vllm-vs-production-llama-20260914",
        "captured_by": "benchmarks/r5_compare/capture_inventory.py",
        "vllm": {
            "repo": str(repo),
            "commit": command(["git", "-C", str(repo), "rev-parse", "HEAD"]),
            "status": command(["git", "-C", str(repo), "status", "--short"]),
            "model": file_record(args.vllm_model / "model.safetensors.index.json"),
            "tokenizer": {
                name: file_record(args.vllm_model / name, hash_file=True)
                for name in (
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "chat_template.jinja",
                )
            },
            "configuration": {
                "target_device": "rocm",
                "arch": "gfx1201",
                "tp": 1,
                "kv": "TurboQuant K8/V4",
                "mtp": 2,
                "adaptive_verification": False,
                "compilation_mode": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "max_num_seqs": 1,
                "max_num_batched_tokens": 256,
                "prefix_caching": True,
                "r5_continuation_hook": "diagnostic only",
            },
        },
        "llama_cpp_production_reference": {
            "container": args.llama_container,
            "container_state": inspect_state,
            "config": safe_config,
            "host_config": safe_host,
            "image_repo_digests": image_config,
            "image_labels": (image_labels),
            "binary_sha256": (
                "055ebdc323ee48e201f0dbb09e83898ebbb2b5fa016aaa3a36edb940cc617568"
            ),
            "version": command(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--entrypoint",
                    "/phase97/bin/llama-server",
                    image or "phase97-candidate-complete:20260906",
                    "--version",
                ]
            ),
            "model": file_record(args.llama_model, hash_file=True),
            "mmproj": file_record(args.llama_mmproj, hash_file=True),
            "draft_model": file_record(args.llama_draft, hash_file=True),
            "source_provenance": {
                "binary_source": "container image; source commit not embedded",
                "image_upstream_revision": (
                    image_labels.get("org.opencontainers.image.revision")
                    if isinstance(image_labels, dict)
                    else None
                ),
                "observed_source_tree": {
                    "path": "/home/emmett/src/llama.cpp-phase0-profile",
                    "commit": command(
                        [
                            "git",
                            "-C",
                            "/home/emmett/src/llama.cpp-phase0-profile",
                            "rev-parse",
                            "HEAD",
                        ]
                    ),
                    "status": command(
                        [
                            "git",
                            "-C",
                            "/home/emmett/src/llama.cpp-phase0-profile",
                            "status",
                            "--short",
                        ]
                    ),
                    "matches_binary": False,
                },
            },
        },
        "gpu": {
            "rocm_smi": command(["rocm-smi"]),
            "nvidia_smi": command(["nvidia-smi"]),
        },
        "comparison_constraints": {
            "same_model": False,
            "same_quantization": False,
            "production_service_modified": False,
            "simultaneous_model_load": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
