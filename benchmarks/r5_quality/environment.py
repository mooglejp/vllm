# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture the existing model environment without loading the diagnostic hook."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path

import flash_attn
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    args = parser.parse_args()
    paths = [
        args.launcher,
        Path("/dev/shm/benchmark_gfx1201_r5_model_ab_hook.py"),
        Path("/dev/shm/r5_model_ab/sitecustomize.py"),
        Path("/dev/shm/r5-flash-site/flash_attn-2.8.3.dist-info/RECORD"),
        Path("/model/config.json"),
        Path("/model/generation_config.json"),
        Path("/model/tokenizer_config.json"),
    ]
    data = {
        "base": "d5fec6675f832ba6bf753247acd092ef86e5cd56",
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "flash_attn": importlib.metadata.version("flash_attn"),
        "flash_file": flash_attn.__file__,
        "forward_file": inspect.getfile(flash_attn.flash_attn_varlen_func),
        "image": (
            "sha256:3cead535c32c11c59b222bbaa502076b9c737fb969b70b7bc9bb8ce77da5698e"
        ),
        "sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
            if p.exists()
        },
        "limits": {
            name: Path("/sys/fs/cgroup", name).read_text().strip()
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
            "profiler_started": False,
            "production_changes": False,
        },
    }
    with args.output.open("x") as output:
        json.dump(data, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
