# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight cgroup resource log; terminate this server on a new OOM."""

import argparse
import json
import os
import signal
import time
from pathlib import Path

from run import check_oom, resources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    args = parser.parse_args()
    initial = resources()
    with args.output.open("x") as output:
        while not args.stop_file.exists():
            row = resources()
            row["host_meminfo"] = Path("/proc/meminfo").read_text()
            stat = os.statvfs("/dev/shm")
            row["shm_used_bytes"] = (stat.f_blocks - stat.f_bfree) * stat.f_frsize
            row["vram"] = {
                str(p): p.read_text().strip()
                for p in Path("/sys/class/drm").glob("card*/device/mem_info_vram_used")
            }
            output.write(json.dumps(row) + "\n")
            output.flush()
            try:
                check_oom(initial, row)
            except RuntimeError:
                if args.pid_file.exists():
                    os.kill(int(args.pid_file.read_text()), signal.SIGTERM)
                raise
            time.sleep(10)


if __name__ == "__main__":
    main()
