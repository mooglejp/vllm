# R5 vLLM / production llama.cpp comparison

Date: 2026-09-15
Comparison id: `r5-vllm-vs-production-llama-20260915`

## Conclusion first

The current vLLM production-equivalent path is the slow baseline, not the R5
candidate. For a 32K cold prompt, the measured median client-side TTFT was:

| path | 32K TTFT | interpretation |
| --- | ---: | --- |
| vLLM baseline | 133.370 s | current slow path |
| vLLM R5 candidate | 61.769 s | 2.16x faster than baseline |
| production llama.cpp | 43.265 s | 1.43x faster than the candidate |

Thus R5 is a substantial vLLM improvement, but it does not beat the current
llama.cpp runtime on this GPU and workload. The existing R5 result of
`131.796 -> 60.643 s` (2.1733x) remains unchanged; this comparison is an
independent client-side reproduction, not a replacement for that record.

This is a runtime comparison, not a vLLM production-adoption decision. No
production default, model, kernel, KV format, or service configuration was
changed.

## Scope and comparability

The measurements were taken sequentially on the same R9700, with no
simultaneous model load. Warmup was separated, prefix reuse was disabled for
the cold speed table, and every speed point used five valid samples and a
fixed 256 output-token limit with EOS ignored.

The two systems are not the same model or quantization:

- vLLM: AMD Quark AWQ MXFP4, TurboQuant K8/V4, TP1, MTP2, adaptive
  verification disabled, compilation mode 0, `FULL_DECODE_ONLY`, one request,
  256-token chunk budget, R5 diagnostic continuation hook enabled.
- llama.cpp: production `Huihui-Qwen3.8-27B-abliterated-UD-Q4_K_XL.gguf`,
  `q8_0` K / `q4_0` V, Vulkan production command, MTP draft mode with two
  draft tokens, and `--reasoning on`.

The exact command, image digest, binary/model hashes, observed source
provenance, and vLLM configuration are in
[`inventory.json`](artifacts/gfx1201_r5_runtime_compare_20260915/inventory.json).
The prepared suite and hashes are in
[`prepared/manifest.json`](artifacts/gfx1201_r5_runtime_compare_20260915/prepared/manifest.json).

Because the model, quantization, tokenizer/template behavior, and reasoning
setting differ, quality numbers below describe the observed deployments; they
are not a controlled claim that one underlying model is intrinsically more
accurate.

## Cold speed

The common metric is median client-side TTFT from request send to the first
generated content. Raw samples and warmup records are retained below
[`results/`](artifacts/gfx1201_r5_runtime_compare_20260915/results/).

| prompt tokens | llama.cpp | vLLM baseline | vLLM R5 candidate | candidate / baseline | candidate / llama.cpp |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.514 s | 0.594 s | 0.594 s | 1.00x | 1.16x |
| 1,024 | 1.566 s | 1.972 s | 1.886 s | 1.05x | 1.20x |
| 4,096 | 4.995 s | 7.260 s | 6.090 s | 1.19x | 1.22x |
| 16,384 | 20.052 s | 44.102 s | 26.684 s | 1.65x | 1.33x |
| 32,768 | 43.265 s | 133.370 s | 61.769 s | 2.16x | 1.43x |

The candidate therefore improves the long-context vLLM path materially, but
the current llama.cpp deployment remains faster at every tested prompt length.
The 32K gap is about 18.5 seconds in favor of llama.cpp. The primary five
candidate samples were the one `valid` record plus the four `retry` records;
the separately retained `extra` record is an additional diagnostic sample and
was not used to change the five-sample median.

The client-observed generation rate is also lower for vLLM in this run:
llama.cpp was approximately 50.7--60.3 token/s across the five lengths,
whereas the candidate was approximately 27.2--30.8 token/s. This is a
serving-level observation that includes each runtime's speculative-decoding
and response-delivery behavior; it is not a kernel-only comparison.

## Quality observations

The vLLM context-suite records are the previously saved R5 records, paired
with the current llama.cpp run. They are not a rerun of the vLLM model, and
the historical R5 internal quality gate remains preserved separately.

| task | vLLM baseline | vLLM candidate | llama.cpp current run |
| --- | ---: | ---: | ---: |
| GSM8K correct | 36/64 | 35/64 | 62/64 |
| MMLU correct | 65/80 | 65/80 | 23/80 |
| HumanEval functional judge | 138/164 | 139/164 | 128/164 |

The llama.cpp MMLU result is heavily affected by its retained production
`--reasoning on` behavior and a 64-token per-case limit: 79/80 MMLU responses
ended by length. The vLLM requests used non-thinking mode. The table should
therefore not be used as an apples-to-apples model-quality ranking. The
HumanEval figures use the isolated stdin-connected judge; raw answers and
judge inputs are retained in the results directory.

The vLLM historical candidate-versus-baseline result remains the prior record:
GSM8K 36 -> 35, MMLU 65 -> 65, HumanEval functional 138 -> 139, with the
previous R5 supplemental quality decision still non-adopted. This comparison
does not rewrite that decision.

## 32K information-retention fixture

The fixed three-content x early/middle/late fixture has nine cases. Strict
scoring strips only leading/trailing whitespace and requires exact equality.

| path | strict exact | auxiliary contains-answer |
| --- | ---: | ---: |
| vLLM baseline | 9/9 | 9/9 |
| vLLM R5 candidate | 9/9 | 9/9 |
| llama.cpp current run | 0/9 | 9/9 |

llama.cpp returned an explanation around each answer, so it fails the strict
one-line fixture while containing the requested value in all nine cases. This
is recorded as an output-format difference, not as evidence that the runtime
failed to retrieve the 32K information. The fixture is intentionally limited
and is not a general long-context quality guarantee.

## Resource and operational notes

- The vLLM comparison container was constrained to 16 GiB RAM with no swap;
  the llama.cpp production reference was inventoried separately and was not
  modified.
- The two runtimes were never loaded simultaneously for measurement.
- No profiler was started. Raw request records include endpoint errors,
  finish reasons, output counts, resource observations, and (for vLLM) the
  R5 hook counters.
- The first disposable vLLM probe exited with status 137 while a recursive
  auxiliary build path was being diagnosed. It was not used for a reported
  sample; the subsequent bounded container completed the valid measurements.
  An earlier baseline 128-token file contains five HTTP 400 records from an
  invalid cache-salt form and is retained as a harness diagnostic; its valid
  retry is the value in the table.
- Prefix-reuse, fixed-state decode regression, and soak were not rerun as part
  of this cross-runtime comparison. They remain separate validation items.

## Practical reading

For short prompts through 4K, llama.cpp is modestly faster in this setup. For
16K and 32K, R5 makes vLLM much better than its baseline, but llama.cpp still
has the lower TTFT and higher observed generation rate. The candidate's
remaining gap is therefore real at the serving level; it should not be
described as the baseline's 133-second behavior. Any future work should first
separate model/quantization/runtime differences before treating the gap as a
single kernel deficit.

The machine-readable summary is
[`comparison_summary.json`](artifacts/gfx1201_r5_runtime_compare_20260915/results/comparison_summary.json).
