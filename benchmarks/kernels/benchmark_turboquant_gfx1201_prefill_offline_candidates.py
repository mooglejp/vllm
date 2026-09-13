# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay saved P2.2 candidate arithmetic without a model run.

The input is one immutable baseline snapshot and its saved FP64 references.
The BF16 and PV-FP32 paths below are PyTorch replays of the existing
online-softmax precision settings on the same BF16 Q/K/V contract; they do not
invoke the production Triton attention launcher. When requested, the existing
Triton cache reader is used only to verify that candidate prefix dequantization
reproduces the saved FP16 workspace and BF16-rounded prefix.
No production launcher, threshold, dispatch, or later phase is changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

TILE_SIZE = 16
REQUIRED_SNAPSHOT = {
    "query",
    "key_chunk",
    "value_chunk",
    "prefix_key_fp16",
    "prefix_value_fp16",
    "prefix_key_bf16",
    "prefix_value_bf16",
    "kv_cache",
    "block_table",
    "q_positions",
    "k_positions",
    "causal_mask",
    "scale",
    "cached_len",
    "seq_len",
    "attention_output",
}
REQUIRED_REFERENCES = {
    "fp64_precast_prefix_bf16",
    "fp64_bf16_cast_prefix_bf16",
    "actual_sdpa_math",
}


def metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float | bool]:
    actual_double = actual.detach().double()
    reference_double = reference.detach().double()
    delta = actual_double - reference_double
    return {
        "finite": bool(
            torch.isfinite(actual_double).all()
            and torch.isfinite(reference_double).all()
        ),
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "relative_l2": float(delta.norm() / reference_double.norm().clamp_min(1e-30)),
        "equal_fraction": float((actual == reference).double().mean()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                actual_double.reshape(1, -1),
                reference_double.reshape(1, -1),
                dim=1,
            )[0]
        ),
    }


def _require(mapping: dict[str, Any], names: set[str], label: str) -> None:
    missing = names - mapping.keys()
    if missing:
        raise ValueError(f"{label} is missing {sorted(missing)}")


def load_inputs(
    snapshot_path: Path,
    references_path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    snapshot = torch.load(snapshot_path, map_location=device, weights_only=True)
    references = torch.load(references_path, map_location=device, weights_only=True)
    if not isinstance(snapshot, dict) or not isinstance(references, dict):
        raise ValueError("snapshot and references must be mappings")
    _require(snapshot, REQUIRED_SNAPSHOT, "snapshot")
    _require(references, REQUIRED_REFERENCES, "references")
    if snapshot.get("sdpa_backend") != "MATH" or not snapshot.get("sdpa_math_forced"):
        raise ValueError("snapshot must record explicitly forced Math SDPA")
    if references.get("sdpa_backend") != "MATH" or not references.get(
        "sdpa_math_forced"
    ):
        raise ValueError("references must record explicitly forced Math SDPA")
    if snapshot.get("layer") != references.get("layer"):
        raise ValueError("snapshot/reference layer mismatch")
    return snapshot, references


def validate_contract(snapshot: dict[str, Any]) -> dict[str, Any]:
    query = snapshot["query"]
    key_chunk = snapshot["key_chunk"]
    value_chunk = snapshot["value_chunk"]
    prefix_key_fp16 = snapshot["prefix_key_fp16"]
    prefix_value_fp16 = snapshot["prefix_value_fp16"]
    prefix_key_bf16 = snapshot["prefix_key_bf16"]
    prefix_value_bf16 = snapshot["prefix_value_bf16"]
    q_len, num_query_heads, head_dim = query.shape
    cached_len = int(snapshot["cached_len"])
    seq_len = int(snapshot["seq_len"])
    if query.dtype != torch.bfloat16:
        raise ValueError(f"expected BF16 query, got {query.dtype}")
    if key_chunk.dtype != torch.bfloat16 or value_chunk.dtype != torch.bfloat16:
        raise ValueError("raw current K/V must be BF16")
    if key_chunk.shape != value_chunk.shape:
        raise ValueError("raw current K/V shapes differ")
    if key_chunk.shape != (q_len, prefix_key_fp16.shape[1], head_dim):
        raise ValueError("raw current K/V shape does not match Q/prefix")
    expected_prefix = (cached_len, key_chunk.shape[1], head_dim)
    for name, tensor in (
        ("prefix_key_fp16", prefix_key_fp16),
        ("prefix_value_fp16", prefix_value_fp16),
        ("prefix_key_bf16", prefix_key_bf16),
        ("prefix_value_bf16", prefix_value_bf16),
    ):
        if tuple(tensor.shape) != expected_prefix:
            raise ValueError(f"{name} shape {tuple(tensor.shape)} != {expected_prefix}")
    if (
        prefix_key_fp16.dtype != torch.float16
        or prefix_value_fp16.dtype != torch.float16
    ):
        raise ValueError("saved workspace prefix must be FP16")
    if (
        prefix_key_bf16.dtype != torch.bfloat16
        or prefix_value_bf16.dtype != torch.bfloat16
    ):
        raise ValueError("saved contract prefix must be BF16")
    if not torch.equal(prefix_key_fp16.to(torch.bfloat16), prefix_key_bf16):
        raise ValueError("saved prefix_key_bf16 is not the FP16-to-BF16 cast")
    if not torch.equal(prefix_value_fp16.to(torch.bfloat16), prefix_value_bf16):
        raise ValueError("saved prefix_value_bf16 is not the FP16-to-BF16 cast")
    if seq_len != cached_len + q_len:
        raise ValueError("seq_len is not cached_len + q_len")
    q_positions = snapshot["q_positions"]
    k_positions = snapshot["k_positions"]
    causal_mask = snapshot["causal_mask"]
    expected_q = torch.arange(
        cached_len,
        seq_len,
        dtype=q_positions.dtype,
        device=q_positions.device,
    )
    expected_k = torch.arange(
        seq_len,
        dtype=k_positions.dtype,
        device=k_positions.device,
    )
    if not torch.equal(q_positions, expected_q) or not torch.equal(
        k_positions, expected_k
    ):
        raise ValueError("saved position vectors do not match the continuation")
    expected_mask = k_positions[None, :] <= q_positions[:, None]
    if not torch.equal(causal_mask, expected_mask):
        raise ValueError("saved causal mask does not match positions")
    if num_query_heads % key_chunk.shape[1] != 0:
        raise ValueError("query/KV head grouping is not integral")
    return {
        "q_len": int(q_len),
        "num_query_heads": int(num_query_heads),
        "num_kv_heads": int(key_chunk.shape[1]),
        "head_dim": int(head_dim),
        "cached_len": cached_len,
        "seq_len": seq_len,
        "scale": float(snapshot["scale"]),
    }


def assemble_contract(
    snapshot: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    cached_len = int(snapshot["cached_len"])
    key = torch.cat((snapshot["prefix_key_bf16"], snapshot["key_chunk"]), dim=0)
    value = torch.cat((snapshot["prefix_value_bf16"], snapshot["value_chunk"]), dim=0)
    if key.shape[0] != cached_len + snapshot["query"].shape[0]:
        raise ValueError("assembled BF16 contract has an unexpected sequence length")
    return key, value


def online_replay_precast(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal_mask: torch.Tensor,
    scale: float,
    *,
    qk_precision: str,
    pv_precision: str,
) -> torch.Tensor:
    """Replay existing online-softmax precision variants before output cast."""
    q_len, num_query_heads, head_dim = query.shape
    num_kv_heads = key.shape[1]
    group = num_query_heads // num_kv_heads
    q_grouped = query.reshape(q_len, num_kv_heads, group, head_dim).permute(1, 0, 2, 3)
    k_grouped = key.permute(1, 0, 2)
    v_grouped = value.permute(1, 0, 2)
    if qk_precision == "fp64":
        softmax_dtype = torch.float64
        q_math = q_grouped.to(softmax_dtype)
        k_math = k_grouped.to(softmax_dtype)
    else:
        softmax_dtype = torch.float32
        q_math = q_grouped.to(torch.bfloat16).to(softmax_dtype)
        k_math = k_grouped.to(torch.bfloat16).to(softmax_dtype)
    pv_dtype = torch.float64 if pv_precision == "fp64" else torch.float32
    v_math = v_grouped.to(pv_dtype)
    e_max = torch.full(
        (num_kv_heads, q_len, group),
        -float("inf"),
        dtype=softmax_dtype,
        device=query.device,
    )
    e_sum = torch.zeros_like(e_max)
    acc_dtype = torch.float64 if pv_precision == "fp64" else torch.float32
    acc = torch.zeros(
        num_kv_heads,
        q_len,
        group,
        head_dim,
        dtype=acc_dtype,
        device=query.device,
    )
    qk_scale = scale * 1.4426950408889634
    for start in range(0, key.shape[0], TILE_SIZE):
        end = min(start + TILE_SIZE, key.shape[0])
        scores = (
            torch.matmul(q_math, k_math[:, start:end].transpose(-1, -2)[:, None])
            * qk_scale
        )
        visible = causal_mask[:, start:end]
        scores = scores.masked_fill(~visible[None, :, None, :], -float("inf"))
        next_max = torch.maximum(e_max, scores.amax(dim=-1))
        next_max = torch.where(
            next_max > -float("inf"), next_max, torch.zeros_like(next_max)
        )
        p = torch.exp2(scores - next_max[..., None])
        alpha = torch.exp2(e_max - next_max)
        e_sum = e_sum * alpha + p.sum(dim=-1)
        acc = acc * alpha[..., None]
        if pv_precision == "fp64":
            p_math = p.to(torch.float64)
        elif pv_precision == "fp32":
            p_math = p.to(torch.float32)
        else:
            p_math = p.to(torch.bfloat16).to(torch.float32)
        acc = acc + torch.matmul(p_math, v_math[:, start:end][:, None])
        e_max = next_max
    return (
        (acc / e_sum.clamp_min(1e-30)[..., None])
        .permute(1, 0, 2, 3)
        .reshape(q_len, num_query_heads, head_dim)
    )


def gate_result(
    result: dict[str, float | bool],
    baseline: dict[str, float | bool],
    factor: float,
) -> dict[str, bool | float]:
    max_limit = float(baseline["max_abs"]) * factor
    rmse_limit = float(baseline["rmse"]) * factor
    max_pass = float(result["max_abs"]) <= max_limit
    rmse_pass = float(result["rmse"]) <= rmse_limit
    finite_pass = bool(result["finite"])
    return {
        "factor": factor,
        "baseline_max_abs": float(baseline["max_abs"]),
        "baseline_rmse": float(baseline["rmse"]),
        "max_abs_limit": max_limit,
        "rmse_limit": rmse_limit,
        "max_abs_pass": max_pass,
        "rmse_pass": rmse_pass,
        "finite_pass": finite_pass,
        "pass": bool(max_pass and rmse_pass and finite_pass),
    }


def prefix_verification(
    snapshot: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Run the existing Triton cache reader and compare its prefix values."""
    if device.type != "cuda":
        raise ValueError("--verify-prefix requires a CUDA device")
    from benchmark_turboquant_gfx1201_prefill_numerics import dump_candidate_cache

    decoded_key, decoded_value = dump_candidate_cache(
        snapshot["kv_cache"], snapshot["block_table"], int(snapshot["cached_len"])
    )
    saved_key_fp16 = snapshot["prefix_key_fp16"]
    saved_value_fp16 = snapshot["prefix_value_fp16"]
    saved_key_bf16 = snapshot["prefix_key_bf16"]
    saved_value_bf16 = snapshot["prefix_value_bf16"]
    decoded_key_fp16 = decoded_key.to(torch.float16)
    decoded_value_fp16 = decoded_value.to(torch.float16)
    decoded_key_bf16 = decoded_key_fp16.to(torch.bfloat16)
    decoded_value_bf16 = decoded_value_fp16.to(torch.bfloat16)
    return {
        "method": "existing_dump_candidate_cache",
        "saved_fp16_contract": {
            "key": metrics(decoded_key_fp16, saved_key_fp16),
            "value": metrics(decoded_value_fp16, saved_value_fp16),
        },
        "saved_bf16_contract": {
            "key": metrics(decoded_key_bf16, saved_key_bf16),
            "value": metrics(decoded_value_bf16, saved_value_bf16),
        },
        "all_fp16_equal": bool(
            torch.equal(decoded_key_fp16, saved_key_fp16)
            and torch.equal(decoded_value_fp16, saved_value_fp16)
        ),
        "all_bf16_equal": bool(
            torch.equal(decoded_key_bf16, saved_key_bf16)
            and torch.equal(decoded_value_bf16, saved_value_bf16)
        ),
    }


def _row(
    route: str,
    output_stage: str,
    output: torch.Tensor,
    reference_stage: str,
    reference: torch.Tensor,
    baseline: dict[str, float | bool],
    gate_factor: float,
) -> dict[str, Any]:
    comparison = metrics(output, reference)
    return {
        "route": route,
        "output_stage": output_stage,
        "reference_stage": reference_stage,
        **comparison,
        "numeric_gate": (
            gate_result(comparison, baseline, gate_factor)
            if output_stage == "bf16_final"
            and reference_stage == "fp64_bf16_cast_prefix_bf16"
            else None
        ),
    }


def markdown_table(result: dict[str, Any]) -> str:
    lines = [
        "# P2.2 offline PyTorch candidate-arithmetic replay",
        "",
        f"- snapshot: `{result['snapshot']}`",
        f"- references: `{result['references']}`",
        f"- layer: `{result['layer']}`",
        f"- device: `{result['device']}`",
        "- input contract: BF16 query, BF16-rounded prefix, raw BF16 current K/V",
        "- SDPA reference: explicit Math backend",
        "- candidate execution: PyTorch arithmetic replay; production Triton "
        "attention launcher not invoked",
        "",
        "## Three-route final BF16 table",
        "",
        "| route | max abs vs FP64 pre-cast | RMSE vs FP64 pre-cast | "
        "rel L2 vs FP64 pre-cast | max abs vs FP64 BF16-cast | "
        "RMSE vs FP64 BF16-cast | rel L2 vs FP64 BF16-cast | gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in result["three_route_final_table"]:
        lines.append(
            "| {route} | {pre[max_abs]:.9g} | {pre[rmse]:.9g} | "
            "{pre[relative_l2]:.9g} | {post[max_abs]:.9g} | "
            "{post[rmse]:.9g} | {post[relative_l2]:.9g} | {gate} |".format(
                route=row["route"],
                pre=row["vs_fp64_precast"],
                post=row["vs_fp64_bf16_cast"],
                gate="PASS" if row["numeric_gate"]["pass"] else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            "## Cast-stage rows",
            "",
            "| route | output stage | reference stage | max abs | RMSE | rel L2 |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in result["stage_rows"]:
        lines.append(
            "| {route} | {output_stage} | {reference_stage} | {max_abs:.9g} | "
            "{rmse:.9g} | {relative_l2:.9g} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Prefix verification",
            "",
            "```json",
            json.dumps(result["prefix_verification"], indent=2),
            "```",
            "",
            "This table is a PyTorch precision-setting replay, not a rerun of "
            "the production Triton candidate kernel. The numeric gate is the "
            "existing synthetic criterion: candidate "
            "max-abs and RMSE must each be no more than 1.1x the old SDPA "
            "error against the same BF16-input FP64 reference after the final "
            "BF16 cast. This artifact closes the diagnostic supplement only; "
            "it is not a production adoption decision.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--save-outputs", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gate-factor", type=float, default=1.10)
    parser.add_argument("--verify-prefix", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    snapshot, references = load_inputs(args.snapshot, args.references, device)
    contract = validate_contract(snapshot)
    key, value = assemble_contract(snapshot)
    query = snapshot["query"]
    causal_mask = snapshot["causal_mask"]
    old_sdpa = references["actual_sdpa_math"]
    if not torch.equal(old_sdpa, snapshot["attention_output"]):
        raise ValueError("saved old SDPA does not match snapshot attention_output")
    fp64_precast = references["fp64_precast_prefix_bf16"]
    fp64_cast = references["fp64_bf16_cast_prefix_bf16"]
    if fp64_precast.dtype != torch.float64 or fp64_cast.dtype != torch.bfloat16:
        raise ValueError("saved BF16-input references have unexpected dtypes")
    if fp64_cast.shape != old_sdpa.shape or fp64_precast.shape != old_sdpa.shape:
        raise ValueError("saved references have unexpected shapes")

    candidate_bf16_precast = online_replay_precast(
        query,
        key,
        value,
        causal_mask,
        contract["scale"],
        qk_precision="bf16",
        pv_precision="bf16",
    )
    candidate_pvfp32_precast = online_replay_precast(
        query,
        key,
        value,
        causal_mask,
        contract["scale"],
        qk_precision="bf16",
        pv_precision="fp32",
    )
    outputs = {
        "old_sdpa": {"bf16_final": old_sdpa},
        "candidate_bf16": {
            "fp32_precast": candidate_bf16_precast,
            "bf16_final": candidate_bf16_precast.to(query.dtype),
        },
        "candidate_pv_fp32": {
            "fp32_precast": candidate_pvfp32_precast,
            "bf16_final": candidate_pvfp32_precast.to(query.dtype),
        },
    }
    baseline_final = metrics(old_sdpa, fp64_cast)
    stage_rows = [
        _row(
            "old_sdpa",
            "bf16_final",
            old_sdpa,
            "fp64_precast_prefix_bf16",
            fp64_precast,
            baseline_final,
            args.gate_factor,
        ),
        _row(
            "old_sdpa",
            "bf16_final",
            old_sdpa,
            "fp64_bf16_cast_prefix_bf16",
            fp64_cast,
            baseline_final,
            args.gate_factor,
        ),
    ]
    for route in ("candidate_bf16", "candidate_pv_fp32"):
        stage_rows.extend(
            [
                _row(
                    route,
                    "fp32_precast",
                    outputs[route]["fp32_precast"],
                    "fp64_precast_prefix_bf16",
                    fp64_precast,
                    baseline_final,
                    args.gate_factor,
                ),
                _row(
                    route,
                    "bf16_final",
                    outputs[route]["bf16_final"],
                    "fp64_bf16_cast_prefix_bf16",
                    fp64_cast,
                    baseline_final,
                    args.gate_factor,
                ),
            ]
        )

    final_rows = []
    for route in ("old_sdpa", "candidate_bf16", "candidate_pv_fp32"):
        final_output = outputs[route]["bf16_final"]
        pre = metrics(final_output, fp64_precast)
        post = metrics(final_output, fp64_cast)
        final_rows.append(
            {
                "route": route,
                "vs_fp64_precast": pre,
                "vs_fp64_bf16_cast": post,
                "numeric_gate": gate_result(post, baseline_final, args.gate_factor),
            }
        )

    if args.verify_prefix:
        prefix_result = prefix_verification(snapshot, device)
    else:
        prefix_result = {
            "method": "not_run",
            "saved_fp16_cast_consistency": {
                "key": bool(
                    torch.equal(
                        snapshot["prefix_key_fp16"].to(torch.bfloat16),
                        snapshot["prefix_key_bf16"],
                    )
                ),
                "value": bool(
                    torch.equal(
                        snapshot["prefix_value_fp16"].to(torch.bfloat16),
                        snapshot["prefix_value_bf16"],
                    )
                ),
            },
            "note": (
                "Run with --verify-prefix on CUDA for exact candidate cache-loader "
                "verification."
            ),
        }

    result = {
        "revision": "91fea1f79eb2bd9e3c6e06ae56b96aa5683bdcda",
        "execution_kind": "offline_torch_operation_replay",
        "production_attention_launcher_executed": False,
        "snapshot": str(args.snapshot),
        "references": str(args.references),
        "layer": snapshot.get("layer"),
        "device": str(device),
        "contract": contract,
        "sdpa_backend": snapshot["sdpa_backend"],
        "sdpa_math_forced": bool(snapshot["sdpa_math_forced"]),
        "candidate_prefix_contract": "saved prefix BF16 plus raw current BF16 K/V",
        "prefix_verification": prefix_result,
        "baseline_old_sdpa_vs_fp64_bf16_cast": baseline_final,
        "three_route_final_table": final_rows,
        "stage_rows": stage_rows,
        "numeric_gate": {
            "criterion": (
                "candidate max_abs and RMSE <= 1.1x old SDPA against BF16-input "
                "FP64 reference after BF16 output cast"
            ),
            "old_sdpa": final_rows[0]["numeric_gate"],
            "candidate_bf16": final_rows[1]["numeric_gate"],
            "candidate_pv_fp32": final_rows[2]["numeric_gate"],
            "diagnostic_completion": True,
            "production_adoption": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown_table(result))
    if args.save_outputs:
        args.save_outputs.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "revision": result["revision"],
                "snapshot": str(args.snapshot),
                "references": str(args.references),
                "layer": snapshot.get("layer"),
                "candidate_bf16_precast": candidate_bf16_precast.detach().cpu(),
                "candidate_bf16_final": outputs["candidate_bf16"]["bf16_final"]
                .detach()
                .cpu(),
                "candidate_pv_fp32_precast": candidate_pvfp32_precast.detach().cpu(),
                "candidate_pv_fp32_final": outputs["candidate_pv_fp32"]["bf16_final"]
                .detach()
                .cpu(),
            },
            args.save_outputs,
        )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
