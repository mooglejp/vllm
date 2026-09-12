# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare single/packed attention numerics on one immutable K8/V4 SoA cache.

This diagnostic does not time kernels or change production launch defaults.
"""

import argparse
import contextlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import torch

from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
from vllm.v1.attention.ops.turboquant_soa import (
    triton_turboquant_decode_gfx1201_k8v4 as specialized,
)
from vllm.v1.attention.ops.turboquant_soa.triton_turboquant_store import (
    triton_turboquant_store,
)
from vllm.v1.attention.ops.turboquant_soa.triton_turboquant_unified_attention import (
    triton_turboquant_unified_attention,
)


class QueryGeometry:
    def __init__(self, kernel, query_block_size, max_query_len, asm_dir=None):
        self.kernel = kernel
        self.query_block_size = query_block_size
        self.max_query_len = max_query_len
        self.asm_dir = asm_dir

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            qbs = self.query_block_size
            blocks = math.ceil(self.max_query_len / qbs)
            kwargs.update(
                QUERY_BLOCK_SIZE=qbs,
                NUM_QUERY_BLOCKS=blocks,
                BLOCK_M=max(16, 2 ** (qbs * 6 - 1).bit_length()),
            )
            grid_override = (grid[0], grid[1], blocks * kwargs["NUM_KV_SPLITS"])
            compiled = self.kernel[grid_override](*args, **kwargs)
            if self.asm_dir is not None:
                name = (
                    f"qb{qbs}_bf16{kwargs['USE_BF16_DOT']}"
                    f"_bs{kwargs['BLOCK_SIZE']}_splits{kwargs['NUM_KV_SPLITS']}"
                )
                destination = self.asm_dir / (name + ".s")
                if not destination.exists():
                    destination.write_text(compiled.asm["amdgcn"])
            return compiled

        return launch


@contextlib.contextmanager
def query_geometry(query_block_size, asm_dir=None):
    """Override only the packed stage-1 geometry in a diagnostic process."""
    original_launch = specialized._launch_multi_token_stage1
    original_kernel = specialized._gfx1201_k8v4_stage1

    def launch(*args, **kwargs):
        kernel = QueryGeometry(
            original_kernel, query_block_size, kwargs["max_query_len"], asm_dir
        )
        with patch.object(specialized, "_gfx1201_k8v4_stage1", kernel):
            return original_launch(*args, **kwargs)

    with patch.object(specialized, "_launch_multi_token_stage1", launch):
        yield


def error_metrics(actual, reference):
    delta = actual.float() - reference.float()
    return {
        "max_abs": delta.abs().max().item(),
        "mean_abs": delta.abs().mean().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "relative_l2": (delta.norm() / reference.float().norm()).item(),
        "equal_fraction": (actual == reference).float().mean().item(),
    }


def evaluate(query, cache, block_table, seq_len, splits, asm_dir=None):
    qlen, num_heads, dim = query.shape
    device = query.device
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    qsl_cpu = torch.tensor([0, qlen], dtype=torch.int32)
    qsl = qsl_cpu.to(device)
    common = dict(
        kv_cache=cache,
        block_table=block_table,
        scale=dim**-0.5,
        max_num_kv_splits=splits,
    )
    outputs = {}
    single_rows = []
    visible_rows = []
    one_qsl_cpu = torch.tensor([0, 1], dtype=torch.int32)
    one_qsl = one_qsl_cpu.to(device)
    for index in range(qlen):
        visible = torch.tensor(
            [seq_len - qlen + index + 1], dtype=torch.int32, device=device
        )
        row = query[index : index + 1]
        row_args = dict(
            query=row,
            seq_lens=visible,
            query_start_loc=one_qsl,
            query_start_loc_cpu=one_qsl_cpu,
            **common,
        )
        single_rows.append(
            specialized.triton_turboquant_decode_gfx1201_k8v4(**row_args)
        )
        with query_geometry(1, asm_dir):
            visible_rows.append(
                specialized.triton_turboquant_decode_gfx1201_k8v4_multi_token(
                    **row_args, max_query_len=qlen
                )
            )
    outputs["single"] = torch.cat(single_rows)
    outputs["per_row_qb1"] = torch.cat(visible_rows)
    for qbs in [1, 2, 4]:
        with query_geometry(qbs, asm_dir):
            outputs[f"packed_qb{qbs}"] = (
                specialized.triton_turboquant_decode_gfx1201_k8v4_multi_token(
                    query=query,
                    seq_lens=seq_lens,
                    query_start_loc=qsl,
                    query_start_loc_cpu=qsl_cpu,
                    max_query_len=qlen,
                    **common,
                )
            )
    cfg = TurboQuantConfig.from_cache_dtype("turboquant_k8v4", dim)
    unused = torch.empty(1, dtype=torch.float32, device=device)
    outputs["generic"] = triton_turboquant_unified_attention(
        query=query,
        kv_cache=cache,
        block_table=block_table,
        seq_lens=seq_lens,
        query_start_loc=qsl,
        Pi=unused,
        centroids=unused,
        scale=dim**-0.5,
        mse_bits=cfg.key_mse_bits,
        key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits,
        value_packed_size=cfg.value_packed_size,
        key_fp8=True,
        max_query_len=qlen,
        max_seq_len=seq_len,
        num_kv_splits=splits,
        tile_size=16,
    )
    results = []
    for name, output in outputs.items():
        assert torch.isfinite(output).all()
        if name == "single":
            continue
        for index in range(qlen):
            results.append(
                dict(
                    mode=name,
                    row=index,
                    visible_seq_len=seq_len - qlen + index + 1,
                    vs_single=error_metrics(output[index], outputs["single"][index]),
                    vs_packed_qb4=error_metrics(
                        output[index], outputs["packed_qb4"][index]
                    ),
                )
            )
    return results


def evaluate_snapshot(path: Path, asm_dir=None):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "block_table",
        "cache",
        "query",
        "query_start_loc",
        "scale",
        "seq_len",
        "splits",
    }
    missing = required - saved.keys()
    if missing:
        raise ValueError(f"{path} is missing snapshot fields: {sorted(missing)}")

    query, cache, block_table = (
        saved[key].cuda() for key in ("query", "cache", "block_table")
    )
    expected_scale = query.shape[-1] ** -0.5
    if saved["scale"] != expected_scale:
        raise ValueError(
            f"{path} has scale {saved['scale']}, expected {expected_scale}"
        )
    record = {
        "snapshot": path.name,
        "query_shape": list(query.shape),
        "seq_len": saved["seq_len"],
        "splits": saved["splits"],
        "positions": saved.get("positions"),
        "metrics": evaluate(
            query,
            cache,
            block_table,
            saved["seq_len"],
            saved["splits"],
            asm_dir,
        ),
    }
    if "output" in saved:
        qsl_cpu = saved["query_start_loc"]
        seq_lens = torch.tensor(
            [saved["seq_len"]], dtype=torch.int32, device=query.device
        )
        replay = specialized.triton_turboquant_decode_gfx1201_k8v4_multi_token(
            query,
            cache,
            block_table,
            seq_lens,
            qsl_cpu.to(query.device),
            saved["scale"],
            query_start_loc_cpu=qsl_cpu,
            max_num_kv_splits=saved["splits"],
            max_query_len=query.shape[0],
        )
        record["replay_vs_live"] = error_metrics(replay.cpu(), saved["output"])
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asm-dir", type=Path)
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[128, 129, 160, 161, 1040, 1041, 3088, 3089],
    )
    parser.add_argument("--splits", type=int, nargs="+", default=[1, 32])
    parser.add_argument("--dtypes", nargs="+", default=["bfloat16", "float16"])
    parser.add_argument("--query-scales", type=float, nargs="+", default=[1.0, 4.0])
    parser.add_argument(
        "--snapshots",
        type=Path,
        nargs="+",
        help="Replay saved real-Q/KV snapshots instead of generated inputs",
    )
    args = parser.parse_args()
    if args.asm_dir:
        args.asm_dir.mkdir(parents=True, exist_ok=True)
    if args.snapshots:
        with args.output.open("x") as results_file:
            for path in args.snapshots:
                record = evaluate_snapshot(path, args.asm_dir)
                encoded = json.dumps(record)
                results_file.write(encoded + "\n")
                results_file.flush()
                print(encoded, flush=True)
        return
    cfg = TurboQuantConfig.from_cache_dtype("turboquant_k8v4", 256)
    with args.output.open("x") as results_file:
        for dtype_name in args.dtypes:
            dtype = getattr(torch, dtype_name)
            for seq_len in args.seq_lens:
                torch.manual_seed(1201 + seq_len)
                key = torch.randn(seq_len, 4, 256, dtype=dtype, device="cuda")
                value = torch.randn_like(key)
                query = torch.randn(3, 24, 256, dtype=dtype, device="cuda")
                for block_size in [16, 32]:
                    num_blocks = math.ceil(seq_len / block_size)
                    block_table = torch.arange(
                        num_blocks - 1, -1, -1, dtype=torch.int32, device="cuda"
                    )[None]
                    positions = torch.arange(seq_len, device="cuda")
                    slots = (
                        block_table[0, positions // block_size] * block_size
                        + positions % block_size
                    )
                    cache = torch.zeros(
                        num_blocks,
                        block_size,
                        4,
                        cfg.slot_size_aligned,
                        dtype=torch.uint8,
                        device="cuda",
                    )
                    unused = torch.empty(1, dtype=torch.float32, device="cuda")
                    triton_turboquant_store(
                        key=key,
                        value=value,
                        kv_cache=cache,
                        slot_mapping=slots,
                        PiT=unused,
                        midpoints=unused,
                        mse_bits=cfg.key_mse_bits,
                        key_packed_size=cfg.key_packed_size,
                        value_quant_bits=cfg.effective_value_quant_bits,
                        key_fp8=True,
                    )
                    for splits in args.splits:
                        for query_scale in args.query_scales:
                            metrics = evaluate(
                                query * query_scale,
                                cache,
                                block_table,
                                seq_len,
                                splits,
                                args.asm_dir,
                            )
                            record = dict(
                                dtype=dtype_name,
                                seq_len=seq_len,
                                block_size=block_size,
                                splits=splits,
                                query_scale=query_scale,
                                metrics=metrics,
                            )
                            encoded = json.dumps(record)
                            results_file.write(encoded + "\n")
                            results_file.flush()
                            print(encoded, flush=True)


if __name__ == "__main__":
    main()
