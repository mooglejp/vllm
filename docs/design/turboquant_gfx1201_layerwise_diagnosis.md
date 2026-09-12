# gfx1201 MTP: the remaining 3K numerical difference

Date: 2026-09-12. The preceding diagnosis and immutable-cache harness were
committed as `232547a6dc`. Production model and kernel code remains unchanged
from `6de0a3e852` throughout this follow-up.

Accuracy follow-up (2026-09-12): the
[small accuracy gate](turboquant_gfx1201_accuracy_gate.md) finds no aggregate
quality regression in 104 seeded reasoning, code, and MMLU cases, extends
target-only B/C equality to 768 full-logit vectors, and adds four real-QKV
attention replays.

## Result

The residual beginning at output token 26 in the previous controlled 3K run
starts in **layer 20's post-attention RMSNorm**, not its GDN update. Layer
indices are zero-based; output-token positions below are one-based.

Identical RMSNorm operands produce different FP32 reductions when processed as
one row versus three rows. This changes two BF16 output elements. Replaying
that operation alone reproduces both live model outputs exactly. In the
controlled MTP run, serializing just that normalization removes the remaining
difference: all vocabulary logits match the serial control at all 40 positions.

This is a numerical execution-shape diagnosis, not a production change or an
accuracy evaluation. The original MTP and non-MTP configurations are still
not output-equivalent.

## Conditions and controls

Hardware, checkpoint, quantization, and prompting are unchanged:
gfx1201 / ROCm 7.2; `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`; MXFP4 emulation;
BF16 MTP weights; eager V2 runner; forced SDPA prefill; two speculative tokens;
adaptive verification disabled. Only the existing 3072-token prompt is used,
with 40 forced output tokens from the original non-MTP trace.

Raw target logits are saved before forcing. The entire actual prefix is checked
against the trace, and rows after incorrect draft proposals are excluded.
Original verification proposals are replayed, and actual input batches are
audited. These are conditional predictions, not free-running generations.

Two comparisons are kept separate:

- **Residual control:** non-MTP uses the common GDN implementation, 2096-token
  physical pages, 16-token kernel blocks, and 2096 + 976 prefill chunks. MTP
  uses the same prefill and existing single-token attention kernel separately
  for each visible row. This reproduces the previous unresolved residual.
- **Target-only replay:** the MTP model is neither constructed nor executed.
  Recorded proposals replace the drafter, while the target retains speculative
  metadata, query batching, and cache geometry. This is not an independently
  implemented scalar reference or ordinary non-speculative scheduling.

For the residual control, target-only replay and actual MTP have identical
input batches and bitwise-identical full logits at all 40 positions. Model
loading reports 17.91 GiB without the MTP model versus 18.74 GiB with it.
This excludes a required contribution from executing the draft model in this
case; it does not by itself prove correctness of a shared target kernel.

### Native A/B/C comparison

The three variants were also rerun without the per-row attention or norm
controls. All use TP=1 and the same 40 validated output prefixes.

| Variant | MTP model executed | Target attention | Prefill tokens |
| --- | --- | --- | --- |
| A: ordinary non-MTP | No | Single-token, kernel block 32, physical page 2080 | 2080 + 992 |
| B: target-only verification replay | No | Production packed kernel, block 16, page 2096 | 3072 |
| C: actual MTP, identical proposal replay | Yes | Production packed kernel, block 16, page 2096 | 3072 |

| Comparison | Full-logit vectors bitwise equal | Argmax equal | Max abs difference | Max KL |
| --- | ---: | ---: | ---: | ---: |
| A versus B | 0/40 | 38/40 | 5.125 | 0.7836168007 |
| B versus C | 40/40 | 40/40 | 0 | 0 |

B/C input batches also match exactly. A/B argmax disagreements are at output
positions 17 and 21; minimum top-ten overlap is 7. Thus `A != B` and `B == C`
hold for this prompt and configuration. B deliberately preserves the target's
verification shape; it is not an independent serial implementation. The
separate residual controls below supply the serial comparison and localize its
last remaining difference. Neither result constitutes a task-accuracy pass.

## Locating the first difference

The serial rerun reproduces the previous top-five values at all 40 positions.
At token 26, embedding and layer 7/15 outputs agree, but layer 23 differs.
Detailed captures narrow the first difference to layer 20's post-attention
RMSNorm. Layers 16-19 and layer 20's input norm, GDN input projections, and
GDN output projection are bitwise identical. Earlier saved checkpoints through
output token 25 also agree.

| Stage at token 26 | Max absolute difference | RMS difference |
| --- | ---: | ---: |
| Layer 20 GDN output | 0 | 0 |
| Layer 20 post-attention norm, normalized output | 0.001953125 | 0.000030517578125 |
| Layer 20 post-attention norm, residual output | 0 | 0 |
| Layer 20 MLP output | 0.0087890625 | 0.0022177953 |
| Final normalized hidden state | 2.0625 | 0.3309301196 |
| Full vocabulary logits | 1.951171875 | 0.3484844863 |

Only coordinates 3846 and 4381 of the 5120-element normalized output change.
The error grows downstream through the quantized model. Cosine, RMS, and
elementwise statistics are retained with the raw intermediate tensors.

Before aligning the norm, the 15 later full-logit vectors differ by maxima
between 1.640625 and 3.5. Conditional argmax still matches at all 40 positions,
but maximum `KL(serial || MTP)` is 0.11937236 and top-ten overlap ranges from
8 to 10. Argmax agreement alone does not characterize the distribution difference.

## RMSNorm-only reproduction

Qwen's `GemmaRMSNorm` computes `weight.float() + 1`. The `vllm_c` implementation
requires matching input/weight dtypes, so BF16 activations with this FP32 weight
use the native PyTorch IR implementation under the configured priorities.
Relevant sources are `vllm/model_executor/layers/layernorm.py`,
`vllm/kernels/vllm_c.py`, and `vllm/ir/ops/layernorm.py`.

The reproduction uses the exact captured GDN output and incoming residual,
the checkpoint's layer-20 norm weight, and epsilon `1e-6`. Operands are checked
bitwise against both model runs. Identical copies of the affected row then
isolate row count from token content.

| Identical rows | FP32 variance | Changed BF16 elements versus one row |
| --- | ---: | ---: |
| 1 | 0.7362732291221619 | 0 |
| 2 | 0.7362731695175171 | 2 |
| 3 | 0.7362731695175171 | 2 |
| 4 | 0.7362731695175171 | 2 |
| 8 | 0.7362732291221619 | 0 |

The variance changes by one FP32 ULP, approximately `5.96e-8`. Batched and
serial native reproductions match their respective live outputs exactly.
Against an FP64 formula, both have the same worst absolute error, 0.0141008749,
over the three captured rows. Their BF16 disagreement does not establish that
one implements a different normalization formula.

The model-level ablation starts from the **residual control**, not native MTP.
It changes only layer 20's post-attention norm to invoke its existing operation
one row at a time during decode. Prefill, GDN, attention, and proposals remain
as in that control. All 40 full-logit vectors then match the serial control
exactly. This is not proposed as a serving implementation or performance fix.

## Excluded observations

- An initial activation probe also ran on the embedding shared with the draft
  model and stopped on a shape assertion. It was corrected to exclude draft
  execution; that incomplete run is excluded.
- A state probe that created GPU temporaries changed the MTP prefill output
  before the residual under study. That run is excluded from layer attribution.
  Removing state extraction restored the original full logits. The allocation
  sensitivity of that probe has not been independently explained.
- The actual recurrent-state cache was FP32. The initially considered BF16
  state round-trip hypothesis does not explain this result. No GDN arithmetic
  change was needed for the successful norm-only ablation.

## Reproduction and remaining gates

The standalone numerical harness accepts the captured operand bundle or
generates seeded inputs. It requires no model load and performs no timing:

```bash
docker exec -e PYTHONPATH=/workspace/vllm -w /workspace/vllm tq-e2e-current \
  /tmp/tq-mtp-eval/.venv/bin/python \
  benchmarks/kernels/benchmark_gemma_rms_norm_numerics.py \
  --input-snapshot /tmp/tq-greedy-diag/norm-replay/input.pt
```

The saved-input replay and seeded BF16/FP16 smoke runs cover row counts
1, 2, 3, 4, and 8. For a seeded run, replace `--input-snapshot ...` with
`--dtype bfloat16 --seed 1201` or `--dtype float16 --seed 1201`.
The changed files pass pre-commit, including Ruff, Markdown lint, and mypy.

Local model instrumentation and exact launch variants are in
`.tools/tq_greedy_diag/`. These controls are restricted to eager, single-request
diagnostics. New full-logit, layer, and kernel-replay data are preserved under
`/tmp/tq-layerwise-diagnosis.WXSGp5`.

The earlier attention-only matrix and this controlled model comparison provide
different evidence and should not be conflated. Accuracy acceptance need not
require bitwise equivalence across unlike execution shapes, but broader prompt
coverage, distribution/ranking metrics, and task evaluations remain necessary.
No default enablement, performance tuning, or accuracy-gate release is made.
