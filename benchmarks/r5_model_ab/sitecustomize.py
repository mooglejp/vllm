# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Install the opt-in R5 diagnostic hook in parent and spawned model workers."""

import os

if os.environ.get("R5_MODE") in {"baseline", "candidate"}:
    import benchmark_gfx1201_r5_model_ab_hook  # noqa: F401
