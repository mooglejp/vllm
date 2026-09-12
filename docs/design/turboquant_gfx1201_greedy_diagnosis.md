# gfx1201 TurboQuant MTP greedy-output diagnosis

Date: 2026-09-11. Source: `6de0a3e8526164282082fa64d24e58ccfe9361c5`.
This investigation changes no production kernel, default, or tuning parameter.

## Result

The original mismatch is not explained by a tiny final-logit margin. However,
controlled execution-path changes eliminate the measured mismatch:

- Disable the **non-MTP GDN packed recurrent decoder**, selecting the same
  recurrent implementation family used for speculative queries.
- Evaluate each MTP target attention row with the existing single-token kernel,
  using that row's visible sequence length to choose split boundaries.
- Match kernel block size, physical page size, and prefill chunk boundaries.

With these controls, all vocabulary logits are **bitwise identical at all 21
saved positions** across three contexts. Conditional argmax IDs also match at
all 120 checked generation positions. Top-five IDs/values match at 105 positions;
the 3K case still has top-five differences at output positions 26-40. Full logits
were not saved at those later positions, and that residual remains unresolved.
Original MTP proposals and verification batches are replayed unchanged; only
the 3K prefill is deliberately split to match the non-MTP control.

This isolates reproducible execution-path differences in these cases, rather
than demonstrating a universal correctness guarantee. It does **not** make the
original configurations output-equivalent or establish task accuracy. In
particular, the row-by-row diagnostic is not a proposed performance fix.

## Model and method

The model is `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`, with the main decoder using
Quark MXFP4 emulation and the checkpoint-excluded MTP tensors loaded in BF16.
The environment is gfx1201 / ROCm 7.2, TP=1, V2 runner, eager execution,
`turboquant_k8v4`, opt-in gfx1201 decode, forced SDPA prefill, and two draft
tokens with adaptive verification disabled. Generic kernel warmup is disabled.
Model length and scheduler token budget are 4096; maximum requests is four,
but diagnostic requests are serialized. RunAI uses CPU staging.

The 128/1024/3072-token prompts and original no-MTP token traces are unchanged
from the earlier evaluation. For each request, the sampler hook:

1. Copies unmodified target logits to CPU FP32 before sampling or forcing.
2. Checks the actual input tokens against the entire common prefix. Rows after
   an incorrect draft proposal are excluded from matched-prefix comparisons.
3. Forces the next token from the original no-MTP trace, producing 40 checked
   conditional positions per context.

Full logits are saved within three positions of each original first mismatch.
These are teacher-forced conditional-logit comparisons, not free-running
generation or performance measurements. The diagnostic client verifies that
the returned tokens equal the forced trace.

## 1. Identical Q/KV, single versus packed

`benchmarks/kernels/benchmark_turboquant_gfx1201_numerics.py` uses one immutable
SoA cache, noncontiguous physical block indices, Hq=24, Hk=4, D=256, three query
rows, BF16/FP16, block sizes 16/32, split counts 1/32, and query scales 1/4.
Lengths are 128, 129, 160, 161, 1040, 1041, 3088, and 3089: 128 cases total.
Each packed row is compared with a single-token launch at its visible length.

| Dtype | Splits | Worst max absolute error | Worst row mean absolute error |
| --- | --- | --- | --- |
| BF16 | 1 | 0 | 0 |
| BF16 | 32 | 0.0078125 | 0.0000762691 |
| FP16 | 1 | 0 | 0 |
| FP16 | 32 | 0.0009765625 | 0.0000096003 |

Packed query widths 1/2/4 are bitwise identical throughout the matrix. Calling
the packed implementation separately for each visible row also exactly matches
the legacy single-token implementation. The observed difference is associated
with shared versus per-row split boundaries, not query width.

Generated BF16 assembly uses `v_wmma_f32_16x16x16_bf16` for both tested logical
row geometries. Logical `BLOCK_M=32` does not by itself establish different
arithmetic results from `BLOCK_M=16`.

Three real layer-63 Q/KV snapshots were also replayed. Single and packed widths
1/2/4 matched on those particular inputs. A second capture including live
attention output verified that compact, byte-preserving cache extraction and
replay reproduce the live output exactly. This does not cover every layer or
every split boundary. The synthetic sweep separately exercises split boundaries;
the model-level control covers all target layers in these requests.

## 2. Raw logits at the original mismatch

Positions below are one-based output-token positions. IDs are tokenizer IDs.

| Prompt | Position | No-MTP top1 / top2 | No-MTP margin | MTP top1 / top2 | MTP margin | Max abs logit difference | Mean abs difference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | 32 | 4203 / 84567 | 0.25 | 20843 / 4203 | 0.125 | 2.495850 | 0.288057 |
| 1024 | 16 | 20843 / 318 | 0.25 | 4203 / 20843 | 0.375 | 2.156250 | 0.290449 |
| 3072 | 17 | 58768 / 18092 | 0.125 | 18092 / 58768 | 0.625 | 3.562500 | 0.493389 |

The 3K comparison additionally includes unequal prefill chunking, described
below. These are not ties separated by just a few thousandths.

## 3. Query-width ablation

Widths 1 and 2 replay all original MTP input batches and proposals exactly.
Both produce bitwise-identical full logits to width 4 at all 21 saved positions.
Conditional argmax results across all 40 positions per context are unchanged.
There is no measured numerical benefit from changing the query width here.

## 4. Cache geometry and prefill

Native non-MTP uses 2080-token physical pages and 32-token kernel blocks;
native MTP uses 2096-token pages and 16-token kernel blocks. A diagnostic
post-model-load alignment hook gives non-MTP 2096-token pages and 16-token
kernel blocks while retaining `fast=true, soa=true` before cache population.
For 128/1024 prompts, all 14 saved logits are unchanged from native non-MTP.

The initial attempt to request `--block-size 2096` directly selected AoS
fallback during model initialization. That run (`single-b16`) is invalid as a
geometry-only comparison and is excluded from the conclusions.

| Mode | 3072-token prefill batches |
| --- | --- |
| Native non-MTP | 2080 + 992 |
| Aligned non-MTP | 2096 + 976 |
| Native MTP | 3072 |
| Final controlled MTP | 2096 + 976 |

Scheduler Mamba alignment and Eagle block-drop handling explain the batching
difference. TurboQuant continuation prefill reads the earlier prefix from
quantized cache; one-shot prefill uses raw K/V. Changing this boundary is not
merely changing cache addresses. Consequently, a 3K comparison that leaves
prefill chunking unequal cannot isolate decode numerics.

## 5. Generic SoA target attention

Only the 16 target attention layers are replaced with generic SoA unified
attention. Draft attention remains specialized. All prefill, proposal, and
verification input batches match the original MTP run exactly.

| Variant | First conditional argmax mismatch: 128 / 1024 / 3072 |
| --- | --- |
| Specialized, query width 4 | 32 / 16 / 17 |
| Specialized, query width 1 | 32 / 16 / 17 |
| Specialized, query width 2 | 32 / 16 / 17 |
| Generic SoA target | 18 / 4 / 21 |

Generic attention changes the divergence positions but does not restore
equivalence. At the original mismatch positions, its max absolute logit
differences from native non-MTP are 1.984375 / 1.875 / 3.640625. Thus neither
"specialized alone is responsible" nor "attention numerics are irrelevant"
follows from this ablation.

## Additional controls

`VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=0` changes non-MTP GDN decode to the
recurrent implementation family used for MTP. This alone makes all seven saved
1K logits exactly equal to MTP. For the 128-token prompt, saved output positions
29-32 also become exact; positions 33-35 still differ, with max errors
2.125 / 2.2890625 / 2.3125.

Keeping GDN aligned and using per-row visible lengths for MTP target attention
eliminates those remaining 128-token differences. Matching the 3K prefill split
eliminates the difference in the saved long-context window too. The final
comparison has zero max/mean error at all 21 saved full-logit vectors and
identical argmax IDs at all 120 conditional positions. The later 3K top-five
residual described above prevents a claim of full numerical equivalence.

The GDN implementations differ in normalization/reduction details, while the
TurboQuant split partition changes the reduction order. Large downstream logit
differences despite small attention perturbations are consistent with
amplification through a quantized model; this investigation does not separately
measure the contribution of each activation-quantization operation.

## Reproduction and scope

The numerical sweep was run in the prepared `tq-rocm-ab` container:

```bash
docker exec -e PYTHONPATH=/workspace/vllm -w /workspace/vllm tq-rocm-ab \
  /tmp/tq-mtp-fixes/.venv/bin/python \
  benchmarks/kernels/benchmark_turboquant_gfx1201_numerics.py \
  --output /tmp/tq-numerics-full.jsonl --asm-dir /tmp/tq-numerics-asm
```

Local model instrumentation is in `.tools/tq_greedy_diag/`: `run_server.sh`
records exact launch flags; `client.py`, `hooks.py`, `compare.py`, and
`replay_attention.py` implement the controls. The isolated Python environments
were created with `uv venv --system-site-packages`. No checkpoint or production
source was edited for these experiments. These hooks are deliberately eager,
single-request diagnostics and are unsuitable for serving or timing.

Raw results, assembly, snapshots, and a copy of the local experiment package
are preserved under `/tmp/tq-greedy-diagnosis.uQW3lH`. Performance tuning and
default enablement remain deferred. Accuracy evaluation and a broader prompt
suite are still needed before accepting the original fast configuration.
