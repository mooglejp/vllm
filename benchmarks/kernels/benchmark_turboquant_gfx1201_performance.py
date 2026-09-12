# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time fixed-buffer gfx1201 packed attention on saved real-model inputs."""

import argparse
import contextlib
import json
import math
import random
import statistics
from pathlib import Path
from unittest.mock import patch

import torch

from vllm.v1.attention.ops.turboquant_soa import (
    triton_turboquant_decode_gfx1201_k8v4 as specialized,
)


class KernelGeometry:
    def __init__(
        self,
        kernel,
        query_block_size: int,
        max_query_len: int,
        tile_size: int,
        num_warps: int,
        num_stages: int,
        artifact_dir: Path | None,
    ) -> None:
        self.kernel = kernel
        self.query_block_size = query_block_size
        self.max_query_len = max_query_len
        self.tile_size = tile_size
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.artifact_dir = artifact_dir

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            qbs = self.query_block_size
            query_blocks = math.ceil(self.max_query_len / qbs)
            kwargs.update(
                QUERY_BLOCK_SIZE=qbs,
                NUM_QUERY_BLOCKS=query_blocks,
                BLOCK_M=max(16, 2 ** (qbs * 6 - 1).bit_length()),
                TILE_SIZE=self.tile_size,
                num_warps=self.num_warps,
                num_stages=self.num_stages,
            )
            adjusted_grid = (
                grid[0],
                grid[1],
                query_blocks * kwargs["NUM_KV_SPLITS"],
            )
            compiled = self.kernel[adjusted_grid](*args, **kwargs)
            if self.artifact_dir is not None:
                stem = (
                    f"qbs{qbs}_tile{self.tile_size}_warps{self.num_warps}"
                    f"_stages{self.num_stages}"
                )
                asm_path = self.artifact_dir / f"{stem}.s"
                if not asm_path.exists():
                    asm_path.write_text(compiled.asm["amdgcn"])
                    metadata = {
                        name: getattr(compiled.metadata, name)
                        for name in (
                            "name",
                            "num_warps",
                            "num_stages",
                            "shared",
                            "n_regs",
                            "n_spills",
                        )
                        if hasattr(compiled.metadata, name)
                    }
                    (self.artifact_dir / f"{stem}.json").write_text(
                        json.dumps(metadata, indent=2) + "\n"
                    )
            return compiled

        return launch


@contextlib.contextmanager
def kernel_geometry(config: dict, max_query_len: int, artifact_dir: Path | None):
    proxy = KernelGeometry(
        specialized._gfx1201_k8v4_stage1,
        config["query_block_size"],
        max_query_len,
        config["tile_size"],
        config["num_warps"],
        config["num_stages"],
        artifact_dir,
    )
    with patch.object(specialized, "_gfx1201_k8v4_stage1", proxy):
        yield


def error_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    delta = actual.float() - reference.float()
    return {
        "max_abs": delta.abs().max().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "relative_l2": (delta.norm() / reference.float().norm()).item(),
        "equal_fraction": (actual == reference).float().mean().item(),
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def benchmark_snapshot(
    path: Path,
    configs: list[dict],
    warmups: int,
    samples: int,
    flush_bytes: int,
    seed: int,
    artifact_dir: Path | None,
) -> list[dict]:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    query, cache, block_table = (
        saved[key].cuda() for key in ("query", "cache", "block_table")
    )
    qsl_cpu = saved["query_start_loc"]
    qsl = qsl_cpu.cuda()
    seq_lens = torch.tensor([saved["seq_len"]], dtype=torch.int32, device="cuda")
    qlen, num_query_heads, head_size = query.shape
    num_kv_heads = cache.shape[2]
    flush = torch.empty(flush_bytes // 4, dtype=torch.float32, device="cuda")

    buffers = {}
    for splits in {config["splits"] for config in configs}:
        buffers[splits] = (
            torch.empty_like(query),
            torch.empty(
                qlen,
                num_query_heads,
                splits,
                head_size + 1,
                dtype=torch.float32,
                device="cuda",
            ),
            torch.empty(qlen, num_query_heads, dtype=torch.float32, device="cuda"),
        )

    def run(config: dict) -> torch.Tensor:
        output, mid, lse = buffers[config["splits"]]
        return specialized.triton_turboquant_decode_gfx1201_k8v4_multi_token(
            query,
            cache,
            block_table,
            seq_lens,
            qsl,
            saved["scale"],
            query_start_loc_cpu=qsl_cpu,
            output=output,
            mid_o_buf=mid,
            lse_buf=lse,
            max_num_kv_splits=config["splits"],
            max_query_len=qlen,
        )

    for config in configs:
        with kernel_geometry(config, qlen, artifact_dir):
            for _ in range(warmups):
                run(config)
    torch.accelerator.synchronize()

    order = list(configs)
    random.Random(seed + saved["seq_len"]).shuffle(order)
    results = []
    for config in order:
        starts = [torch.Event(enable_timing=True) for _ in range(samples)]
        ends = [torch.Event(enable_timing=True) for _ in range(samples)]
        with kernel_geometry(config, qlen, artifact_dir):
            for start, end in zip(starts, ends):
                flush.zero_()
                start.record()
                output = run(config)
                end.record()
        torch.accelerator.synchronize()
        times_us = [start.elapsed_time(end) * 1000 for start, end in zip(starts, ends)]
        correctness = error_metrics(output.cpu(), saved["output"])
        query_blocks = math.ceil(qlen / config["query_block_size"])
        cache_bytes = (
            query_blocks
            * saved["seq_len"]
            * num_kv_heads
            * specialized.LOGICAL_BYTES_PER_SLOT
        )
        partial_bytes = qlen * num_query_heads * config["splits"] * (head_size + 1) * 4
        output_bytes = qlen * num_query_heads * (head_size * query.element_size() + 4)
        estimated_bytes = cache_bytes + 2 * partial_bytes + output_bytes
        median_us = statistics.median(times_us)
        results.append(
            {
                "snapshot": path.name,
                "seq_len": saved["seq_len"],
                "query_shape": list(query.shape),
                **config,
                "samples": samples,
                "flush_bytes": flush_bytes,
                "median_us": median_us,
                "p95_us": percentile(times_us, 0.95),
                "min_us": min(times_us),
                "estimated_min_bytes": estimated_bytes,
                "estimated_gb_s": estimated_bytes / median_us / 1000,
                "vs_live": correctness,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshots", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-block-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--splits", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--tile-sizes", type=int, nargs="+", default=[16])
    parser.add_argument("--num-warps", type=int, nargs="+", default=[4])
    parser.add_argument("--num-stages", type=int, nargs="+", default=[1])
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--flush-mib", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args()
    if args.artifact_dir is not None:
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
    configs = [
        {
            "query_block_size": query_block_size,
            "splits": splits,
            "tile_size": tile_size,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
        for query_block_size in args.query_block_sizes
        for splits in args.splits
        for tile_size in args.tile_sizes
        for num_warps in args.num_warps
        for num_stages in args.num_stages
    ]
    with args.output.open("x") as output:
        for path in args.snapshots:
            for result in benchmark_snapshot(
                path,
                configs,
                args.warmups,
                args.samples,
                args.flush_mib * 1024 * 1024,
                args.seed,
                args.artifact_dir,
            ):
                encoded = json.dumps(result)
                output.write(encoded + "\n")
                output.flush()
                print(encoded, flush=True)


if __name__ == "__main__":
    main()
