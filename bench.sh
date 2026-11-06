#!/usr/bin/env bash
set -euo pipefail

# P-core to pin to (confirmed via /sys/devices/cpu_core/cpus)
PCORE=0

# Generations per Criterion iteration; divide perf counts by GENS_RUN
export GOL_GENS="${GOL_GENS:-1000}"

export RUSTFLAGS="-Ctarget-cpu=native"

# Locate the bench executable Cargo built
BENCH=$(cargo bench --no-run --message-format=json 2>/dev/null \
  | jq -r 'select(.reason == "compiler-artifact" and (.target.kind | index("bench"))) | .executable' \
  | tail -n1)

if [[ -z "$BENCH" ]]; then
  echo "could not find bench executable" >&2
  exit 1
fi

EVENTS="cpu_core/LLC-loads/u,cpu_core/LLC-load-misses/u,cpu_core/instructions/u,cpu_core/cycles/u"

for side in 20 30; do
  echo "== side $side =="
  taskset -c "$PCORE" perf stat -e "$EVENTS" \
    "$BENCH" --bench "clock/$side" 2>&1
  echo
done