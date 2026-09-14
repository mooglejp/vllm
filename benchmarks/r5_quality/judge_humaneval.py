# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute HumanEval candidates in locked-down disposable containers."""

import argparse
import json
import subprocess
import uuid
from pathlib import Path

import regex as re


def _load(path: Path) -> list[dict]:
    return [
        row
        for line in path.read_text().splitlines()
        if (row := json.loads(line))["task"] == "humaneval"
    ]


def _candidate_source(row: dict) -> str:
    text = row["text"].strip()
    fences = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.I)
    if fences:
        text = fences[-1].strip()
    prompt = row["reference"]["prompt"]
    entry_point = row["reference"]["entry_point"]
    if re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", text):
        return text
    return prompt + text + "\n"


def _judge(row: dict, image: str, venv: Path, python_root: Path) -> dict:
    source = _candidate_source(row)
    script = (
        source
        + "\n"
        + row["reference"]["test"]
        + "\ncheck("
        + row["reference"]["entry_point"]
        + ")\n"
    )
    container_name = "r5-humaneval-" + uuid.uuid4().hex
    command = [
        "docker",
        "run",
        "--interactive",
        "--rm",
        "--name",
        container_name,
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=32m",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--memory",
        "512m",
        "--memory-swap",
        "512m",
        "--cpus",
        "1",
        "--pids-limit",
        "64",
        "--ulimit",
        "nofile=64:64",
        "--user",
        "1000:1000",
        "--mount",
        f"type=bind,source={venv},target=/judge/.venv,readonly",
        "--mount",
        f"type=bind,source={python_root},target={python_root},readonly",
        "--entrypoint",
        "/judge/.venv/bin/python",
        image,
        "-I",
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            input=script,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
        passed = completed.returncode == 0
        detail = completed.stderr[-2000:]
    except subprocess.TimeoutExpired:
        passed = False
        detail = "timeout"
        subprocess.run(
            ["docker", "kill", container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    return {"id": row["id"], "passed": passed, "detail": detail}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", default="rocm/vllm-dev:base")
    parser.add_argument("--venv", type=Path, default=Path("/home/emmett/vllm-tq/.venv"))
    parser.add_argument(
        "--python-root",
        type=Path,
        default=Path(
            "/home/emmett/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu"
        ),
    )
    args = parser.parse_args()

    rows = _load(args.input)
    results = []
    for index, row in enumerate(rows, 1):
        result = _judge(
            row,
            args.image,
            args.venv.resolve(),
            args.python_root,
        )
        results.append(result)
        print(f"[{index}/{len(rows)}] {result['id']}: {result['passed']}", flush=True)
    summary = {"passed": sum(row["passed"] for row in results), "cases": len(results)}
    with args.output.open("x") as output:
        json.dump({"summary": summary, "cases": results}, output, indent=2)
        output.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
