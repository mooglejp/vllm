# gfx1201 Radiance-delta diagnostic plan

Status: cache-format and continuation-backend controls complete; no production path adopted

Base revision: `a85fec26ba`

This document starts after the long-prefill phase plan closed without an
adopted P2--P5 component. It is a diagnostic control, not a request to reopen
the rejected streaming, W4A8, or raw-attention candidates.

## Scope and constraints

The first control separates cache-format cost from the attention implementation
under one explicit contract. It does not change production defaults, decode,
MTP2, K8/V4 ownership, scheduler policy, or any cache dtype. No Radiance code
is copied; the control is a clean-room implementation of the local tensor
contract.

The GPU run used an AMD Radeon AI PRO R9700 (gfx1201), ROCm `7.2.53211`, and
Torch `2.12.0+git6bbd260` in `tq-e2e-current`. The fixed contract is:

- BF16 Q/K/V, `Hq=24`, `Hk=4`, `D=256`, GQA6;
- one causal continuation request, block size 16;
- raw BF16 current chunk; only the cached prefix representation changes;
- explicit `torch.nn.attention.SDPBackend.MATH` for every attention call;
- no sinks, sliding window, or full score-matrix oracle.

The installed gfx1201 FlashAttention/CK path was already observed to segfault
in `ck_tile::FmhaFwdKernel` during the P4 control. It is therefore not used as
an unverified “efficient” comparator here.

## Measurement contract

`benchmarks/kernels/benchmark_gfx1201_kv_format_diagnostic.py` creates one
seeded logical BF16 Q/K/V tensor set per case and uses three prefix formats:

- `bf16_math`: logical BF16 prefix and current chunk, Math SDPA;
- `fp8_prefix`: prefix stored as FP8 E4M3, decoded to BF16, raw BF16 current
  chunk, Math SDPA;
- `k8v4_prefix`: prefix stored with the existing TurboQuant K8/V4 writer and
  decoded by the existing full-dequant reader, raw BF16 current chunk, Math
  SDPA.

Each format has an attention-only timing on pre-materialized BF16 K/V, a
full-path timing including prefix decode and dense K/V assembly, and a
decode-only timing. Outputs are preallocated, compilation is warmed, samples
use rotating order and a 64 MiB device flush, and all raw samples are saved.
The FP64 oracle evaluates only rows `0`, `floor(q_len/2)`, and `q_len-1` for
heads `0` and `23`; no `q_len x cached_len` score matrix is constructed.

This is a synthetic tensor control. Its FP8 and K8/V4 error values are not a
model-quality result because the K8/V4 metadata uses the benchmark's fixed
reference configuration rather than captured model calibration.

## Results

Median device times are in microseconds. The attention-only columns use the
same Math SDPA implementation; full columns include prefix decode and current
chunk assembly.

| cached / q | BF16 attention | FP8 attention | K8/V4 attention | FP8 full | K8/V4 full | FP8 decode | K8/V4 decode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4K / 256 | 9,191.0 | 9,189.9 | 9,090.7 | 9,263.7 | 9,334.1 | 165.0 | 188.9 |
| 32K / 128 | 53,406.0 | 54,148.6 | 52,978.5 | 55,246.7 | 54,726.0 | 1,284.7 | 1,433.5 |
| 32K / 256 | 77,513.1 | 80,760.8 | 80,206.4 | 76,451.8 | 82,129.7 | 1,353.2 | 1,546.7 |
| 32K / 512 | 148,859.0 | 145,054.3 | 152,334.9 | 154,576.8 | 154,251.0 | 1,340.7 | 1,466.7 |

All outputs were finite. At 32K/q256, representative-row relative L2 error
against the raw BF16 FP64 oracle was `0.03699` for FP8 and `0.11258` for K8/V4;
at 32K/q512 it was `0.04005` and `0.11781`, respectively. The BF16 Math
baseline was about `0.00166`--`0.00168` relative L2. These are format/error
diagnostics only, not an adoption gate or a production-quality claim.

## Continuation backend control (2026-09-13)

The follow-up control kept the same synthetic K8/V4 cache and compared the
current validated K8/V4 direct reader with two attention-only controls:

- `bf16_math_attention_only`: raw BF16 prefix/current K/V with Math SDPA;
- `k8v4_all_quantized_math_attention_only`: every cached K/V position decoded
  to BF16 first, then the same Math SDPA;
- `k8v4_direct_reader_split1`: the existing unified K8/V4 direct-reader
  adapter, with `q_len=128` rows, `max_num_kv_splits=1`, and the same cache
  metadata.

This is a backend control, not a replay of a production continuation request,
and it does not use the rejected P2.2 streaming candidate. The attention inputs
and preallocated outputs are fixed per case; warmup, rotating order, raw
samples, and a 64 MiB flush are retained from the cache-format experiment.

| cached / q | BF16 Math | K8/V4 all-quantized Math | K8/V4 direct reader | direct / BF16 |
| ---: | ---: | ---: | ---: | ---: |
| 4K / 128 | 6,332.4 us | 6,261.5 us | 12,399.5 us | 1.96x |
| 32K / 128 | 52,879.7 us | 52,597.9 us | 95,224.9 us | 1.80x |

The all-quantized Math control is within 1.1% of the raw BF16 Math control,
while the split=1 direct-reader adapter is about 1.8--2.0x slower. This split=1
value is a diagnostic setting, not the production direct-reader latency. Its
outputs were finite and
its representative-row relative L2 errors against the raw BF16 FP64 oracle
were `0.10952` (4K) and `0.12090` (32K); these remain synthetic cache-format
diagnostics, not model-quality results. The corresponding K8/V4 decode-only
times were `261.7 us` and `1,531.4 us`, far below the direct-reader attention
times.

The fixed-cache evidence therefore points at the direct reader's backend or
work partitioning rather than at K8/V4 materialization alone. The same Math
control also shows the expected chunking effect at 32K: q128/q256/q512
attention medians were `52,879.7`/`77,513.1`/`148,859.0 us`, or roughly
`413`/`303`/`291 us` per query token. This is a diagnostic observation only;
it does not authorize a scheduler or production chunk-size change.

## Unified chunk backend control (2026-09-13)

The split setting and query decomposition were then separated in one
benchmark-only run. The cache and K8/V4 metadata stayed fixed, and the q128
query was run through:

- the existing decode adapter with `max_num_kv_splits=1`;
- the same adapter with the production default
  `tq_max_kv_splits_for_cuda_graph=32`;
- `triton_turboquant_unified_attention` directly as one sequence with
  `query_start_loc=[0, 128]`, `seq_lens=[cached_len + 128]`,
  `max_query_len=128`, and `force_2d=True`.

All three K8/V4 paths read the quantized cache for both prefix and current
positions. The final path is therefore a work-partition control for the
current production contract, not the rejected P2.2 raw-current streaming
candidate. Math SDPA and the all-quantized Math control are retained as the
same-case references.

| cached / q | BF16 Math | K8/V4 all-quantized Math | adapter split=1 | adapter split=32 | unified chunk 2D |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4K / 128 | 6,278.0 us | 6,189.3 us | 12,396.6 us | 10,241.5 us | 5,007.4 us |
| 32K / 128 | 53,860.7 us | 53,043.0 us | 95,215.6 us | 76,158.6 us | 38,529.3 us |

The production split improves the adapter by about 1.21x (4K) and 1.25x
(32K) over split=1, so the earlier result must not be called production
latency. The unified chunk path is still 2.05x/1.98x faster than the
production-split adapter and 2.48x/2.47x faster than split=1. It is also
1.25x/1.40x faster than the BF16 Math control at these two lengths.

All outputs were finite. Against the representative-row raw BF16 FP64 oracle,
the unified path had relative L2 `0.10951` (4K) and `0.12088` (32K), matching
the direct-reader envelope (`0.10952`/`0.12090`). Its representative-row
differences from the production-split adapter had max-abs `0.0004883`/`0.0001221`
and relative L2 `0.00285`/`0.00304`. These are synthetic numerical controls,
not model-quality gates.

This result supports the hypothesis that converting a q128 continuation into
128 one-token decode requests is a major cost, while retaining K8/V4 itself is
not. It is not yet a production adoption result: the benchmark uses one
synthetic request, no sinks or sliding window, and no model replay. A future
production experiment would need a default-off opt-in, actual metadata, model
quality/capacity checks, and a cold long-prefill A/B; no dispatch or threshold
was changed here.

## q-length scope control (2026-09-13)

Before wiring an opt-in, the same control swept q16/q32/q64/q128 at 4K and
32K. The table shows the production-split adapter and unified 2D medians:

| cached / q | adapter split=32 | unified chunk 2D | unified speedup |
| ---: | ---: | ---: | ---: |
| 4K / 16 | 1,517.5 us | 4,740.8 us | 0.32x |
| 4K / 32 | 2,694.5 us | 4,806.9 us | 0.56x |
| 4K / 64 | 5,141.8 us | 4,899.3 us | 1.05x |
| 4K / 128 | 9,898.1 us | 4,936.0 us | 2.00x |
| 32K / 16 | 10,064.9 us | 37,267.5 us | 0.27x |
| 32K / 32 | 18,970.7 us | 36,561.8 us | 0.52x |
| 32K / 64 | 39,629.8 us | 37,881.4 us | 1.05x |
| 32K / 128 | 76,473.0 us | 38,635.0 us | 1.98x |

All unified outputs in this sweep were finite. The crossover makes a blanket
q_len<=128 replacement unjustified: q16/q32 regress, and q64 is below the
1.3x exploratory speed margin. The first production candidate is therefore
limited to cached `q_len == 128` behind the new default-off
`VLLM_TQ_GFX1201_K8V4_UNIFIED_CONTINUATION` opt-in. The existing q<128 adapter,
q>128 continuation path, first-chunk prefill, decode, MTP2, and cache layout
remain unchanged. The model-quality and cold-32K gates are still pending.

## q128 model opt-in gate (2026-09-13)

The q128-only opt-in was replayed on the official Quark MXFP4 model with TP1,
TurboQuant K8/V4, MTP2, adaptive verification disabled, `max_num_batched_tokens`
128, compilation disabled with `FULL_DECODE_ONLY`, one request, and fixed greedy
64-token output. The baseline and candidate used the same generated token
prompts and server settings; only
`VLLM_TQ_GFX1201_K8V4_UNIFIED_CONTINUATION` changed. A first candidate 4K
request included Triton compilation, so it was not used for the speed gate. A
second request in the same candidate process was used after compilation:

| model case | baseline TTFT | candidate TTFT | speedup | token output |
| --- | ---: | ---: | ---: | --- |
| 4K / q128 continuation | 13.220 s | 11.691 s | 1.13x | 47/64 token IDs differ |

The candidate was finite and the server completed normally, but it missed the
1.5x exploratory model gate and changed the greedy output substantially. The
32K candidate was intentionally not run after this gate failure. The q128
unified continuation opt-in is therefore **rejected and remains disabled**;
the result does not reopen P2.2 or change the production q<128 adapter,
q>128 continuation, decode, MTP2, or cache layout.

Artifacts are `/tmp/tq-radiance-delta-20260913/model-q128/model-q128-baseline.jsonl`,
`/tmp/tq-radiance-delta-20260913/model-q128/model-q128-candidate.jsonl`, and
`/tmp/tq-radiance-delta-20260913/model-q128/model-q128-candidate-4k-repeat.jsonl`.
The reproducible request harness is
`benchmarks/benchmark_gfx1201_unified_continuation_model.py`.

## Interpretation and stop point

At 32K/q256, prefix decode was `1.353 ms` for FP8 and `1.547 ms` for K8/V4,
versus `77.513 ms` for the common Math attention. At q512 the corresponding
figures were `1.341 ms`, `1.467 ms`, and `148.859 ms`. Thus, under this fixed
Math-SDPA control, replacing K8/V4 with FP8 reduces the cache decode portion
slightly but does not remove the dominant attention cost. Cache format alone
does not explain the long-continuation slowdown observed in the corrected P4
trace.

Both controls are complete and do not justify an FP8-KV production opt-in or a
direct-reader dispatch change. No model rerun, cache-layout change, attention
kernel, scheduler, or dispatch change follows from these numbers. Any future
continuation optimization must use a materially different backend/runtime
hypothesis and pass separate correctness, availability, and performance gates
on gfx1201.

Artifacts:

- [benchmark source](/home/emmett/vllm-tq/benchmarks/kernels/benchmark_gfx1201_kv_format_diagnostic.py)
- `/tmp/tq-radiance-delta-20260913/kv-format-4k32k.json`
- `/tmp/tq-radiance-delta-20260913/kv-format-32k-q512.json`
- `/tmp/tq-radiance-delta-20260913/kv-format-direct-reader.json` (historical
  split=1 run)
- `/tmp/tq-radiance-delta-20260913/kv-format-unified-chunk.json`
- `/tmp/tq-radiance-delta-20260913/kv-format-unified-sweep.json`
- `/tmp/tq-radiance-delta-20260913/kv-format-smoke.json`

Reproduction inside the GPU container:

```bash
/tmp/tq-venv/bin/python \
  benchmarks/kernels/benchmark_gfx1201_kv_format_diagnostic.py \
  --output /tmp/tq-kvdiag-20260913.json \
  --cached-lens 4096 32768 --q-lens 128 256 \
  --warmups 2 --samples 3 --flush-mib 64

/tmp/tq-venv/bin/python \
  benchmarks/kernels/benchmark_gfx1201_kv_format_diagnostic.py \
  --output /tmp/tq-kvdiag-20260913-q512.json \
  --cached-lens 32768 --q-lens 512 \
  --warmups 2 --samples 3 --flush-mib 64

/tmp/tq-venv/bin/python \
  benchmarks/kernels/benchmark_gfx1201_kv_format_diagnostic.py \
  --output /tmp/tq-kvdiag-direct-20260913.json \
  --cached-lens 4096 32768 --q-lens 128 \
  --warmups 2 --samples 3 --flush-mib 64

/tmp/tq-venv/bin/python \
  benchmarks/kernels/benchmark_gfx1201_kv_format_diagnostic.py \
  --output /tmp/tq-kvdiag-unified-20260913.json \
  --cached-lens 4096 32768 --q-lens 128 \
  --warmups 2 --samples 3 --flush-mib 64 \
  --production-max-num-kv-splits 32

/tmp/tq-venv/bin/python \
  benchmarks/kernels/benchmark_gfx1201_kv_format_diagnostic.py \
  --output /tmp/tq-kvdiag-unified-sweep-20260913.json \
  --cached-lens 4096 32768 --q-lens 16 32 64 128 \
  --warmups 2 --samples 3 --flush-mib 64 \
  --production-max-num-kv-splits 32
```
