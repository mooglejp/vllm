# gfx1201 TurboQuant MTP stage-1 pipeline tuning

Date: 2026-09-12. Baseline: `58a9c3244f`. Hardware: gfx1201 / ROCm 7.2.

## Result

Changing only the multi-token stage-1 Triton launch from `num_stages=1` to
`num_stages=2` improves real-QKV kernel latency at medium and long contexts
while preserving every checked output. The 111-token point regresses by 2.5%,
within the existing 5% short-context gate; the 3090-token point is 1.57x faster.

The gain is measurable but small end to end. On the existing fixed-output model
workload, median decode throughput rises by 0.14% to 0.56%. This establishes
that attention is not the main limit in the current eager Quark-emulation run.
Further attention-only tuning is not justified without a new profile showing a
larger share. The route remains opt-in.

## Kernel sweep

The primary sweep uses byte-preserving layer-63 snapshots from the actual MTP
model. Each case has three BF16 query rows, 24 query heads, four KV heads,
head size 256, an immutable K8/V4 SoA cache, and 32 split-K partitions. Output,
partial, and LSE buffers are preallocated. Compilation, allocations, snapshot
loading, transfers, and a 64 MiB L2 flush are outside the timed interval.

FlashInfer's CUPTI helper is not installed in this ROCm environment, so the
initial matrix uses HIP events after five untimed launches and reports the
median of 50 cold-L2 samples. The narrowed confirmation table uses 20 untimed
launches and 300 samples. Candidate order is seeded and shuffled. The final 3K
pair is also traced independently with rocprofv3.

An initial matrix covers query block sizes 1, 2, and 4 and split counts 1, 2,
4, 8, 16, and 32 at real sequence lengths 111, 406, 856, 1777, and 3090.
Production query width 4 and 32 splits is best at 1777 and 3090, and within
5.8% of the best candidate at 406 and 856. At 111, launch overhead dominates;
some smaller geometries are faster but change the split reduction shape.
The production geometry is retained.

The bounded follow-up holds query width 4, tile 16, 32 splits, and four warps
constant, changing only stage count:

| Real sequence length | Stages 1 median | Stages 2 median | Speedup |
| ---: | ---: | ---: | ---: |
| 111 | 102.41 us | 105.06 us | 0.975x |
| 406 | 96.40 us | 93.15 us | 1.035x |
| 856 | 90.90 us | 85.95 us | 1.058x |
| 1777 | 90.84 us | 79.14 us | 1.148x |
| 3090 | 130.32 us | 83.16 us | 1.567x |

Tile 32 changes the reduction order and is slower at 3K. Two warps are also
slower. Tile 64 compilation is disproportionately expensive in this Triton
version; the over-broad exploratory compile is excluded and was terminated
before measurement. No result from that incomplete run influences the choice.

rocprofv3 records 25 calls, including warmups, for each 3K candidate. Stage-1
average duration falls from 130.24 us to 58.66 us; minimum duration falls from
114.72 us to 46.84 us. Profiling adds variance and is not substituted for the
event table, but it independently confirms that the improvement occurs inside
`_gfx1201_k8v4_stage1`, not in Python or allocation overhead.

Generated gfx1201 assembly retains wave32 and
`v_wmma_f32_16x16x16_bf16`. The stages-1 kernel declares 248 VGPRs, 105 SGPRs,
and no private segment. The stages-2 candidate declares 256 VGPRs, 67 SGPRs,
and an eight-byte private segment. The measured pipeline benefit outweighs the
small VGPR/private-segment increase on the target device.

## Correctness after tuning

The launch parameter does not change program semantics, but the full relevant
matrix is rerun after the production edit:

- `tests/quantization/test_turboquant.py::TestGfx1201K8V4Decode`: 26 passed,
  including FP16/BF16, block sizes 16/32, ragged multi-token requests, mixed
  decode/prefill, and HIP graph replay;
- 128 synthetic correctness records and all 1,920 rowwise metrics are identical
  to the pre-tuning artifact;
- four tracked real-QKV replay JSON records are byte-for-byte identical;
- fixed-output model token hashes and per-request MTP acceptance counters are
  unchanged at 128, 1024, and 3072 prompt tokens.

The prior small task-accuracy suite is not rerun because both the synthetic and
real-input kernel outputs, as well as fixed model outputs, are unchanged.

## End-to-end model result

The model workload and environment exactly match the earlier baseline: one
request at a time, greedy sampling, 64 output tokens with EOS ignored, unique
cache salts, one warmup and three measured repeats per point, eager V2 runner,
two MTP draft tokens, adaptive verification disabled, forced SDPA prefill, and
Quark MXFP4 emulation. Rates below are medians.

| Prompt tokens | Decode before | Decode after | Change | E2E before | E2E after | Change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 7.822 | 7.833 | +0.14% | 7.609 | 7.622 | +0.18% |
| 1024 | 8.106 | 8.130 | +0.29% | 7.096 | 7.115 | +0.26% |
| 3072 | 7.396 | 7.437 | +0.56% | 5.225 | 5.248 | +0.43% |

Every measured repeat within a context produces the same token hash before and
after. Accepted draft-token counts per request also remain 37/52, 39/50, and
36/54 respectively. The result therefore compares equal work and equal model
behavior. The before run was recorded on the previous day and GPU clocks were
not fixed, so the sub-percent end-to-end changes are directional rather than a
statistically significant serving-speed claim. The kernel measurements, not
that small model-level delta, are the basis for adopting the launch change.

The current runtime reports that native MXFP4 compute is unavailable and uses
simulated weight dequantization plus activation QDQ with high-precision linear
layers. Those model GEMMs, the drafter, and GDN dominate the approximately
7.4-8.1 decode tok/s result. The 40-60 tok/s project objective cannot be assessed
from this emulation environment, and the older 29.5/17.7 figures from another
runtime remain non-comparable.

The follow-up software-fused decode linear and its new bottleneck profile are
documented in [gfx1201 software-fused MXFP4 decode linear](turboquant_gfx1201_mxfp4_fusion.md).

## Artifacts and decision

The tracked fixed-buffer harness reproduces the stage comparison from a saved
3K snapshot:

```bash
docker exec -e PYTHONPATH=/workspace/vllm:/tmp/tq-venv/lib/python3.12/site-packages \
  -w /workspace/vllm tq-e2e-current \
  /tmp/tq-mtp-eval/.venv/bin/python \
  benchmarks/kernels/benchmark_turboquant_gfx1201_performance.py \
  /tmp/tq-greedy-diag/abc-c-coarse-v2/attention-c3072.pt \
  --output /tmp/tq-stage-confirm.jsonl --query-block-sizes 4 --splits 32 \
  --tile-sizes 16 --num-warps 4 --num-stages 1 2 \
  --warmups 20 --samples 300
```

The exact real snapshots, phase-one geometry sweep, stage confirmation, generated
assembly, rocprofv3 CSV traces, post-change correctness outputs, and fixed-output
model records are preserved under `/tmp/tq-accuracy-eval.ReUkmn`.

The candidate satisfies the existing gate: long-context attention improves,
short-context regression stays below 5%, and correctness is unchanged. Adopt
`num_stages=2` for the exact multi-token gfx1201 K8/V4 path. Do not change split
count, tile size, query width, single-token launch, or default enablement from
this sweep.
