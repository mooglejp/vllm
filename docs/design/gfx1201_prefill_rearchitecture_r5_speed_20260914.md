# R5 evaluation policy v2: upstream tolerance and continuation timing

Start: `66cd1152ab`, isolated branch `codex/gfx1201-prefill-r5`.
The original checkout's user modification to `.gitignore` is preserved.

## Policy fixed before this measurement

The [strict evaluation](gfx1201_prefill_rearchitecture_r5_20260914.md) and its
artifact remain unchanged. Its Math-relative 1.10x gate failed. That result is
a strict reproducibility diagnostic, not rejection of the AMD Triton backend.

This separate policy permits timing after the causal fixture, input contract,
full-output finiteness and upstream-derived tolerance pass. The loaded
FlashAttention 2.8.3 `flash_attn_triton_amd/test.py` uses
`torch.testing.assert_close`, `atol=0.01`, `rtol=0.01`, against
`attention_forward_pytorch_ref_impl`, with FP16 inputs. Here the same formula
`abs(actual-reference) <= 0.01 + 0.01 * abs(reference)` is applied to BF16
candidate versus explicit Math SDPA over identical effective BF16 inputs.
This is a documented BF16/reference adaptation, not a claim to reproduce the
upstream FP16 test exactly. Full-output finiteness is independently required.
The source hashes and precise package paths are in the artifact.

Both paths use the same Q, retained SoA K8/V4 cache decoded through FP16 then
BF16, raw BF16 current K/V, scale 0.0625, and bottom-right causal boundary.
The original boundary suite remains preserved; this run rechecks the fixed
cached3/q2 fixture and the two long timing cases. Inputs are synthetic, not
captured model activations. FP64 reference rows preserve absolute positions.
Differences and reductions are calculated in FP64. The names
`fp64_reference_difference` and `final_bf16_reference_difference` distinguish
the independent unrounded oracle from its BF16-rounded value. Both compare
the kernel's **final BF16 output**; neither observes its internal accumulator.

## Timing boundaries and results

Cache allocation/store and fixed metadata are prepared before timing.
Attention-only includes the public attention API and its output allocations.
Continuation includes decoding existing cache, FP16-to-BF16 conversion,
joining raw current K/V, required mask or cumulative-length metadata, output
allocation and attention. No explicit preparation synchronization remains
inside that interval. Both arms use five warmups, 20 raw samples, alternating
order and a 64 MiB flush. The measurement uses ROCm events rather than CUDA
CUPTI, following the existing ROCm harness.

| Prefix / query | Math continuation median | AMD Triton median | Speedup |
| --- | ---: | ---: | ---: |
| 4096 / 256 | 4.449870 ms | 0.918547 ms | 4.844x |
| 32768 / 256 | 38.099447 ms | 5.261342 ms | 7.241x |

Both cases pass the v2 numerical prerequisites and fail the retained strict
Math-ratio diagnostic. The 32K continuation threshold of 1.30x passes.
These are fixed-input continuation measurements, not model TTFT speedups.
The same R9700 uses Torch 2.12.0+rocm7.14.0 / HIP 7.14.60850, the pinned
FlashAttention 2.8.3 wheel and AMD selector from the strict record.

## Model A/B validation

Model A/B is conditional on the preceding pass. Its outcome is recorded below
once the isolated model verification finishes. The TTFT threshold remains a
10% reduction on cold 32K/chunk256, with at least five alternating samples per
arm. Quality, prefix reuse and operational qualification remain separate.

## Reproduction

Run the following in the same pinned environment, with the isolated
FlashAttention wheel directory on `PYTHONPATH`:

```bash
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
  .venv/bin/python benchmarks/kernels/benchmark_gfx1201_r5_attention_speed.py \
  --output /tmp/r5-attention-speed.json --warmups 5 --samples 20 --flush-mib 64
```

The [machine-readable artifact](artifacts/gfx1201_prefill_rearchitecture_r5_speed_20260914.json)
contains every raw timing sample, full numerical checks, independent FP64
metrics, backend provenance and the policy manifest. No production source,
kernel, cache format or rejected candidate was changed.
