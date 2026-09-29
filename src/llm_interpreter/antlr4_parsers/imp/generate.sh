#!/usr/bin/env bash
set -euo pipefail

# Configuration
RUNS=3000
OUTDIR="train_ds"
SCRIPT="imp_fuzzer_generator.py"
PYTHON_CMD="python"   # change to `python3` if needed

mkdir -p "$OUTDIR"

for i in $(seq 1 "$RUNS"); do
  outfile="$OUTDIR/${i}.imp"
  echo "[$(date +'%F %T')] Running #$i -> $outfile"
  # write stdout to file, preserve exit code; stderr will be printed to console
  if ! $PYTHON_CMD "$SCRIPT" > "$outfile"; then
    echo "ERROR: run #$i failed (exit $?). Stopping." >&2
    exit 1
  fi
done

echo "All $RUNS runs finished. Files in: $OUTDIR"
