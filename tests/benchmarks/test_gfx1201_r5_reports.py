# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for bounded R5 trace parsing and consumer attribution."""

import ast
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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


def test_quality_mode_requires_idle_metrics():
    """Do not switch a module-wide diagnostic mode with pending requests."""
    from benchmarks.r5_quality.run import require_idle

    idle = ["vllm:num_requests_running 0", "vllm:num_requests_waiting 0"]
    require_idle(idle)
    with pytest.raises(RuntimeError):
        require_idle([idle[0], "vllm:num_requests_waiting 1"])
    with pytest.raises(RuntimeError):
        require_idle([])


def test_quality_salts_isolate_modes_and_accept_case_paths():
    from benchmarks.r5_quality.run import cache_salt

    left = cache_salt("suite/baseline/gsm8k/1")
    right = cache_salt("suite/candidate/gsm8k/1")
    assert left != right
    assert len(left) <= 128 and not any(c in left for c in "@/\\\x00")


def test_short_request_clears_previous_hook_counts(tmp_path):
    """Control reset is independent of eligible continuation or GPU execution."""
    source = Path("benchmarks/benchmark_gfx1201_r5_model_ab_hook.py")
    tree = ast.parse(source.read_text())
    functions: list[ast.stmt] = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in {"_refresh_control", "_forward"}
    ]
    control = tmp_path / "control.json"
    control.write_text(json.dumps({"mode": "candidate", "run_id": "short"}))
    counts = {
        "run_id": "previous",
        "calls": 1984,
        "applied_calls": 1984,
        "shapes": {"old": 1},
        "layers": {"old": 1},
        "enabled": True,
    }
    saved = []
    namespace: dict[str, Any] = {
        "os": SimpleNamespace(environ={"R5_MODE_FILE": str(control)}),
        "json": json,
        "Path": Path,
        "_COUNTS": counts,
        "_write_stats": lambda: saved.append(dict(counts)),
        "_ORIGINAL_FORWARD": lambda *a, **k: "unchanged",
        "TurboQuantAttentionImpl": object,
        "Any": object,
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    assert namespace["_forward"](None) == "unchanged"
    assert saved[0]["calls"] == saved[0]["applied_calls"] == 0
    assert saved[0]["shapes"] == saved[0]["layers"] == {}
    assert saved[0]["run_id"] == "short"
    namespace["_forward"](None)
    assert len(saved) == 1
    control.write_text(json.dumps({"mode": "baseline", "run_id": "short"}))
    with pytest.raises(RuntimeError, match="within a run ID"):
        namespace["_forward"](None)


def test_quality_scope_rejects_stale_counts_and_accepts_fresh_zero(tmp_path):
    from benchmarks.r5_quality.run import scope

    prefix = tmp_path / "stats"
    path = tmp_path / "stats.1"
    path.write_text(json.dumps({"run_id": "old", "calls": 16, "applied_calls": 16}))
    with pytest.raises(RuntimeError, match="Missing or ambiguous"):
        scope(prefix, "new", "candidate")
    path.write_text(json.dumps({"run_id": "new", "calls": 0, "applied_calls": 0}))
    assert scope(prefix, "new", "candidate")["coverage"] == "unapplied_regression"
