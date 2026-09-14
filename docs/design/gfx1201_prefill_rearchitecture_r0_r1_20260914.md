# gfx1201 prefill rearchitecture R0/R1 result — 2026-09-14

## Scope and stop condition

This report records the R0 freeze and R1 benchmark-only measurement requested from
`ba57aece9e60d637459556ace36708a9a025cb26` on the isolated worktree
`codex/gfx1201-prefill-r0-r1-local`. The pre-existing `.gitignore` change in the
original worktree was left untouched. No adopted production path, decode path,
K8/V4 cache path, MTP path, threshold, dispatch, R2 MXFP4 fusion, or production
integration was changed.

R1 evaluates only raw FP8 E4M3FN GEMM mappings H1/H2. It intentionally does not
claim that raw FP8 or large-tile/multi-wave GEMM is impossible when this mapping
gate fails.

## R0 freeze

The frozen manifest is
[`r0_manifest`](/home/emmett/vllm-tq/.worktrees/prefill-r1/docs/design/artifacts/gfx1201_prefill_rearchitecture_r0_manifest_20260914.json).
The R0 contract fixes:

- AMD Radeon AI PRO R9700 / `gfx1201`, Torch `2.12.0+rocm7.14.0`, HIP `7.14.60850`;
- raw FP8 E4M3FN `uint8` bytes, pre-expanded inputs, FP64 byte-oracle checks,
  and normalized pre-cast error `max <= 1e-3`, relative L2 `<= 1e-4`;
- effective shapes `(N,K,calls)` = `(5120,6144,64)`, `(34816,5120,64)`,
  `(5120,17408,64)`, `(16384,5120,48)`, `(96,5120,48)`, and
  `(14336,5120,16)`, measured at M=64 and M=256;
- primary scope M=256 with N>=512 and the frozen call-count weighting;
- five warmups, twenty samples, order rotation, fixed buffers, and a 64 MiB
  cache-flush condition.

The old A0 5x gate and P2.2 attention thresholds are not reused.

## R1 implementation and measurement

The benchmark-only extension is
[`gfx1201_prefill_v3.cu`](/home/emmett/vllm-tq/.worktrees/prefill-r1/csrc/rocm/gfx1201_prefill_v3.cu).
The harness is
[`benchmark_gfx1201_prefill_v3.py`](/home/emmett/vllm-tq/.worktrees/prefill-r1/benchmarks/kernels/benchmark_gfx1201_prefill_v3.py).

- H1: 128x64 output tile, four waves, K-slab 64, padded LDS, register
  accumulators, and direct FP32 output. The timed wrapper postcasts to a
  preallocated BF16 output.
- H2: 256x64 output tile, eight waves, with the same K-slab and output contract.
- The old A3 diagnostic mapping is retained as a historical comparison.
- The primary baseline is same-session pre-expanded BF16 `torch.mm`; FP32 `torch.mm`
  remains a diagnostic control. N=96 is measured for fallback coverage but is not
  included in the large-N weighted gate.

No allocation, conversion, compilation, oracle, or correctness work is inside
the timed operation. H1/H2 timed output includes the BF16 postcast so its output
contract matches the BF16 baseline; the retained FP32 scratch is preallocated.

## Correctness result

The R1 artifact contains 17 correctness cases, all passing:

- basis vectors;
- full 129x129x64 row/column-distinct input;
- full 65x129x65 M/N/K-tail input;
- full H1 128x64x64 and H2 256x64x64 tiles;
- all six frozen production-shape families at M=64 and M=256 with representative
  row/column FP64 oracle checks and full-output finiteness.

The detailed JSONL, including raw samples, is
[`r1_measurements`](/home/emmett/vllm-tq/.worktrees/prefill-r1/docs/design/artifacts/gfx1201_prefill_rearchitecture_r1_20260914.jsonl).

## M=256 timing (median microseconds; H1/H2 speedup vs BF16)

| shape | N | K | calls | BF16 | H1 | H2 | H1/BF16 | H2/BF16 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5120 | 6144 | 64 | 317.221 | 333.841 | 319.781 | 0.9502x | 0.9920x |
| 1 | 34816 | 5120 | 64 | 987.743 | 1999.505 | 1395.983 | 0.4940x | 0.7076x |
| 2 | 5120 | 17408 | 64 | 646.702 | 868.862 | 826.882 | 0.7443x | 0.7821x |
| 3 | 16384 | 5120 | 48 | 616.902 | 955.122 | 647.122 | 0.6459x | 0.9533x |
| 4 | 96 | 5120 | 48 | 42.240 | — | — | —x | —x |
| 5 | 14336 | 5120 | 16 | 558.572 | 831.802 | 635.441 | 0.6715x | 0.8790x |

The N>=512 calls-weighted totals are:

| method | weighted latency (us) | speedup vs same-session BF16 | gate |
| --- | ---: | ---: | --- |
| old A3 | 1450039.692 | 0.1127x | fail |
| H1 | 264095.998 | 0.6189x | fail |
| H2 | 203958.232 | 0.8014x | fail |
| BF16 baseline | 163455.042 | 1.0000x | reference |

The required R1 feasibility gate is complete correctness plus at least 1.0x
against same-session BF16 in the M=256, N>=512, calls-weighted scope. H1 and
H2 both fail it. H2 is closer, but remains below parity.

## M=64 diagnostic timing

M=64 is reported independently and is not used to promote an H1/H2 mapping:

| shape | N | K | calls | BF16 | H1 | H2 | H1/BF16 | H2/BF16 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5120 | 6144 | 64 | 254.121 | 316.141 | 391.421 | 0.8038 | 0.6492 |
| 1 | 34816 | 5120 | 64 | 703.242 | 1278.023 | 1833.245 | 0.5503 | 0.3836 |
| 2 | 5120 | 17408 | 64 | 431.741 | 790.322 | 1001.143 | 0.5463 | 0.4312 |
| 3 | 16384 | 5120 | 48 | 381.221 | 611.542 | 820.162 | 0.6234 | 0.4648 |
| 4 | 96 | 5120 | 48 | 39.580 | — | — | — | — |
| 5 | 14336 | 5120 | 16 | 343.541 | 580.901 | 810.322 | 0.5914 | 0.4240 |

N=96 (shape 4) remains a fallback/correctness point; H1/H2 were intentionally not
included in the large-N timing gate there.

## Generated-object observations

The object-level record is
[`r1_object_observation`](/home/emmett/vllm-tq/.worktrees/prefill-r1/docs/design/artifacts/gfx1201_prefill_rearchitecture_r1_object_20260914.txt).
The final validation used the coherent ROCm 7.14 environment and recorded the
host/object/code-object hashes, compiler, resource metadata, and static opcode
counts. H1/H2 use one wave32 per wave, padded LDS staging, FP8 WMMA operations,
and register accumulators; tail variants add LDS scratch traffic for partial
stores. These are observations of the generated artifact, not a causal
performance attribution.

## Decision

R0 and R1 are complete and committed. H1 and H2 are rejected for this R1
feasibility gate because neither reached same-session BF16 parity in the required
weighted scope, despite complete correctness. The result is an architecture-level
measurement of these two mappings only; it does not reject future large-tile or
multi-wave designs with a different mapping hypothesis.

Stop here. Do not start R2, MXFP4 decode/scale fusion, additional tile sweeps,
model execution, or production integration under this R1 result.
