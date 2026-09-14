# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Create the fixed token-ID prompts used by the R5 model A/B client."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base = tokenizer.encode(
        "The following technical note describes a tensor program for "
        "long-context inference. ",
        add_special_tokens=False,
    )
    prompts = {}
    for length, label in ((4096, "smoke4k"), (32768, "cold32k")):
        prompts[label] = (base * ((length + len(base) - 1) // len(base)))[:length]
    args.output.write_text(json.dumps(prompts) + "\n")


if __name__ == "__main__":
    main()
