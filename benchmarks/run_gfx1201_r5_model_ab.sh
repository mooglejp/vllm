#!/usr/bin/env bash
# Run inside the isolated container after the 4K scope smoke has passed.
set -euo pipefail
output_dir=${1:?fresh output directory}
mkdir "$output_dir"
suite_id=$(date +%s%N)
for phase in warmup sample0 sample1 sample2 sample3 sample4; do
  modes=(baseline candidate)
  if [[ "$phase" == sample1 || "$phase" == sample3 ]]; then
    modes=(candidate baseline)
  fi
  for mode in "${modes[@]}"; do
    /tmp/tq-venv/bin/python /dev/shm/benchmark_gfx1201_r5_model_ab.py \
      --prompt-file /dev/shm/r5_prompts.json \
      --output "$output_dir/$phase-$mode.json" \
      --case cold32k --run-id "$suite_id-$phase-$mode" --mode "$mode" \
      --control-file /dev/shm/r5_model_control.json
  done
done
