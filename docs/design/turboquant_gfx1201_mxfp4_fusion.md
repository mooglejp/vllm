# gfx1201 software-fused MXFP4 decode linear

Date: 2026-09-12. Baseline: `605693317e`. Hardware: gfx1201 / ROCm 7.2.

## Result

The Quark emulation path materializes every MXFP4 weight as BF16 before each
linear operation. For small decode batches this repeats a large weight
dequantization and then reads the expanded BF16 matrix into a separate GEMM.
The new `TritonGfx1201Mxfp4LinearKernel` decodes packed E2M1 weights and E8M0
scales into a register tile immediately before a BF16 `tl.dot`, retaining an
FP32 accumulator. It does not materialize the full BF16 weight matrix.

The route is disabled by default and requires
`VLLM_ROCM_USE_GFX1201_MXFP4_GEMM=1`. It is selected only on gfx1201 for
dynamic-MXFP4 activations when `amd-quark` is available for activation QDQ.
Each invocation uses the fused path only when all of the following hold:

- flattened row count is at most four;
- output width is at least 512;
- activation dtype is BF16;
- no bias is present.

Calls outside these limits delegate to `EmulationMxfp4LinearKernel`, including
normal large-M prefill with more than four rows and the model's 96-wide
projection. Dispatch depends on row count, not the prefill/decode phase: a
prefill or chunk tail with one to four rows can use the fused path. Other
activation formats and weight-only configurations retain their existing
backends. Native-MXFP4 platforms continue to select their existing
higher-priority backend.

On the fixed 64-output-token MTP workload, median decode throughput improves
from 7.44--8.13 tok/s to 16.21--18.00 tok/s, a 2.18--2.23x increase. Every
token hash, the aggregate MTP acceptance counters, and all 768 checked full
target-logit vectors remain unchanged.

## Baseline attribution

A PyTorch GPU trace records one 3072-token prefill followed by 32 generated
tokens. Fourteen target-model decode scopes are present because MTP verifies up
to three positions per target invocation. GPU launches are assigned to a target
scope through each launch's runtime correlation ID, rather than by overlapping
asynchronous device timestamps.

Before fusion, correlated target-decode dispatches total 3561.00 ms:

| Operation | Calls | GPU time | Share |
| --- | ---: | ---: | ---: |
| MXFP4 weight dequantization | 4,256 | 1577.67 ms | 44.3% |
| BF16 dense GEMM | 3,584 | 1571.79 ms | 44.1% |
| Activation MXFP4 QDQ | 4,256 | 20.57 ms | 0.6% |
| All other correlated dispatches | -- | 390.98 ms | 11.0% |

Weight dequantization plus the following high-precision GEMM therefore account
for 88.4% of the correlated target device work. The corresponding profiler
annotation reports 4.212 s of target CUDA time. Prefill reports 3.376 s and is
outside the small-row optimization.

## Kernel geometry and original benchmark result

The initially selected launch used a 16x32 output tile, K block 128, two wave32
warps, and one pipeline stage. Packed nibbles and one E8M0 scale per 32 values
are loaded directly. K values that are not multiples of 128 and N values that
are not multiples of 32 are masked. The OCP edge encodings are explicit: raw
E8M0 zero is `2^-127`, while raw 255 is NaN.

The table below came from the original wrapper-level benchmark over the six
dense dimensions observed in the Qwen3.5-27B checkpoint. Inputs, packed weights,
scales, and the 64 MiB L2 flush buffer were fixed before timing, but the
linear wrapper allocated its output tensor inside each HIP-event interval. The
result is still sufficient to establish the large fusion benefit, but it is not
a fixed-output kernel-only measurement suitable for selecting small
launch-configuration differences.

| N x K | M=1 speedup | M=3 speedup | Correctness |
| ---: | ---: | ---: | --- |
| 5120 x 6144 | 2.85x | 2.43x | bitwise exact |
| 34816 x 5120 | 4.27x | 4.16x | bitwise exact |
| 5120 x 17408 | 3.19x | 3.22x | bitwise exact |
| 16384 x 5120 | 4.22x | 4.33x | bitwise exact |
| 96 x 5120 | 1.01x | 1.15x | bitwise exact |
| 14336 x 5120 | 3.79x | 3.79x | bitwise exact |

The 96-wide result is measured to validate the dispatch boundary, but
production deliberately retains emulation there. The fused path is beneficial
for every eligible model shape.

The subsequent
[launch-configuration tuning](turboquant_gfx1201_mxfp4_launch_tuning.md)
separates wrapper timing from direct launches into preallocated outputs,
measures candidates in round-robin order, and updates the production geometry.

The initial generated gfx1201 ISA uses wave32 and no private segment. It
declares 132 live VGPR indices and 105 live SGPR indices; rocprof reports
allocation-rounded counts of 136 VGPRs and 128 SGPRs, zero scratch, and zero
dynamic LDS for the dispatch. The Triton launch metadata reports 4096 bytes of
shared staging.

## End-to-end model result

The before and after runs use one request at a time, greedy sampling, 64 output
tokens with EOS ignored, unique cache salts, one warmup plus three measured
repeats, eager V2 execution, two MTP draft tokens, adaptive verification off,
the opt-in gfx1201 K8/V4 attention path, and forced SDPA prefill. Rates are
medians; the baseline is the immediately preceding `num_stages=2` result.

| Prompt tokens | Decode before | Decode fused | Speedup | E2E before | E2E fused | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 7.833 | 17.463 | 2.23x | 7.622 | 16.159 | 2.12x |
| 1024 | 8.130 | 18.002 | 2.21x | 7.115 | 13.996 | 1.97x |
| 3072 | 7.437 | 16.212 | 2.18x | 5.248 | 8.578 | 1.63x |

All four requests at each context, including warmups, produce the same token
hash as the baseline. Across those 12 requests, both runs record 312 speculative
steps, 624 drafted tokens, and 448 accepted draft tokens. The production route
also reproduces all 768 full-FP32 target-logit SHA-256 values over 12 prompts
and 64 output positions, with all 323 target input batches identical.

The model throughput result agrees with the standalone kernel direction but is
not a native-MXFP4 hardware claim. GPU clocks were not locked. The workload is
single-request eager decoding, so it does not establish throughput under
continuous batching or graph capture.

## Post-fusion profile

The matched optimized trace reduces the 14 target scopes from 4.212 s to
2.071 s of profiler-reported CUDA time. Correlated target dispatches fall from
3561.00 ms to 1007.03 ms, or 3.54x. The fused kernel contributes 765.92 ms over
3,584 calls. Activation QDQ contributes 24.40 ms. The eligible large weight
dequantizations and their 3,584 BF16 GEMMs are replaced by the fused kernel;
the 48 small, 96-wide projections per target forward still use emulation.

Prefill remains effectively unchanged at 3.390 s versus 3.376 s because its row
count selects emulation. The remaining model-level gap comes from prefill,
Python/kernel-launch overhead, the separately executed MTP drafter, GDN, and
the fused kernel itself. Further attention tuning is still not justified by the
profile. The subsequent
[whole-decode profile](turboquant_gfx1201_post_fusion_profile.md) includes the
drafter and target sampling, and confirms the next kernel target using their
shares of total decode device work.

## Correctness and validation

Correctness is established at four levels:

- GPU unit tests compare packed decoding plus GEMM against Quark-style
  dequantization followed by `F.linear`, exactly, for M=1/3/4, masked N/K
  boundaries including N=513, and raw E8M0 values 0, 1, 127, 254, and 255;
- the six real model dimensions at M=1 and M=3 are bitwise exact in the tracked
  benchmark;
- fixed model outputs and MTP acceptance counters match the prior backend;
- 768 full target-logit vectors and 323 input batches match the prior backend.

The new environment switch defaults to false. CPU selection tests cover the
disabled, wrong-architecture, and enabled gfx1201 cases. Existing emulation is
kept as the per-call fallback rather than duplicating its activation QDQ or
weight-loading behavior.

## Reproduction and artifacts

Run the fixed-buffer benchmark on gfx1201 with:

```bash
docker exec \
  -e PYTHONPATH=/workspace/vllm:/tmp/tq-venv/lib/python3.12/site-packages \
  -w /workspace/vllm tq-e2e-current \
  /tmp/tq-mtp-eval/.venv/bin/python \
  benchmarks/kernels/benchmark_mxfp4_gfx1201.py \
  --output /tmp/tq-mxfp4.jsonl --rows 1 3 \
  --warmups 10 --samples 50 --artifact-dir /tmp/tq-mxfp4-asm
```

The exact benchmark JSONL, model records, target-logit hashes, baseline and
optimized model traces, rocprof CSV, generated assembly, and metadata are
preserved under `/tmp/tq-accuracy-eval.ReUkmn`.

Adopt the backend as an opt-in acceleration for software-emulated dynamic
MXFP4 decode on gfx1201. Do not enable it by default until graph-mode,
multi-request, wider-model, and broader serving validation are complete.
