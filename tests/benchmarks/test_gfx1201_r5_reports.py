# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for bounded R5 trace parsing and consumer attribution."""

import gzip
import json

import pytest

from benchmarks.benchmark_gfx1201_r5_trace_summary import events, summarize


def test_only_cpu_scope_launches_attribute_candidate_kernels(tmp_path):
    """GPU annotation copies and launches outside the scope must not inflate counts."""
    path = tmp_path / "trace.json.gz"
    scope = {
        "name": "r5_candidate_continuation",
        "cat": "user_annotation",
        "pid": 11,
        "tid": 12,
        "ts": 10,
        "dur": 5,
    }
    data = [scope, dict(scope, cat="gpu_user_annotation", pid=0, tid=1)]
    for timestamp, correlation, name in ((12, 7, "attn_fwd.kd"), (20, 8, "ck_tile")):
        data.append(
            {
                "cat": "hip_runtime",
                "pid": 11,
                "tid": 12,
                "ts": timestamp,
                "args": {"correlation": correlation},
            }
        )
        data.append(
            {"cat": "kernel", "name": name, "args": {"correlation": correlation}}
        )
    with gzip.open(path, "wt") as output:
        json.dump({"traceEvents": data}, output)
    result = summarize(path)
    assert result["candidate_scope_count"] == 1
    assert result["flash_attn_fwd_count"] == 1
    assert result["correlated_ck_count"] == 0


def test_truncated_array_header_fails_instead_of_waiting_forever(tmp_path):
    path = tmp_path / "broken.json.gz"
    with gzip.open(path, "wt") as output:
        output.write('{"traceEvents":')
    with pytest.raises(ValueError, match="Missing traceEvents array"):
        list(events(path))
