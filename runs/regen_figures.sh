#!/bin/bash
# =============================================================================
# regen_figures.sh — one-command regeneration of every paper figure from committed scripts.
# The executable form of brain/papers/REPRODUCIBILITY.md (the claims->script->output map). Nature/ICLR
# reproducibility: every figure in Paper A + Paper B regenerates from the released checkpoints/data.
#
# CPU/login-runnable analysis (no training, no GPU). Run from the nanoprot/ repo root:
#   bash runs/regen_figures.sh            # regenerate all available figures
#   PAPER=B bash runs/regen_figures.sh    # only Paper B figures
#   PAPER=A bash runs/regen_figures.sh    # only Paper A figures
# Each step is guarded by its input data; missing inputs are reported and skipped (not fatal), so the
# script is runnable on any checkout with a subset of the cached data.
# =============================================================================
set -uo pipefail

REPO="${NANOPROT_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
export NANOPROT_BASE_DIR="${NANOPROT_BASE_DIR:-$REPO/.cache/nanoprot}"
export MPLBACKEND=Agg
TR="$NANOPROT_BASE_DIR/trajectory_results"
PY="${PYTHON:-python}"
PAPER="${PAPER:-AB}"
ok=0; skip=0; fail=0

run() {  # run <fig-label> <input-path-to-check> <command...>
  local label="$1" need="$2"; shift 2
  if [ -n "$need" ] && [ ! -e "$need" ]; then
    echo "  [skip] $label  (missing input: $need)"; skip=$((skip+1)); return
  fi
  if "$@" > "/tmp/regen_${label}.log" 2>&1; then
    echo "  [ok]   $label"; ok=$((ok+1))
  else
    echo "  [FAIL] $label  (see /tmp/regen_${label}.log)"; fail=$((fail+1))
  fi
}

echo "== regenerating figures (NANOPROT_BASE_DIR=$NANOPROT_BASE_DIR) =="
if [[ "$PAPER" == *B* ]]; then
  echo "-- Paper B --"
  run rank_reversal      "$TR" $PY -m scripts.rank_reversal       --results-dir "$TR" --out docs/rank_reversal.csv
  run rank_reversal_fig  "$TR" $PY -m scripts.plot_rank_reversal --results-dir "$TR" --out docs/figures/rank_reversal
  run crossover_scaling  docs/rank_reversal.csv $PY -m scripts.plot_crossover_scaling --csv docs/rank_reversal.csv --out docs/figures/crossover_scaling
  run forecast_v2        "$TR" $PY -m scripts.forecast            --results-dir "$TR" --pythia-csv docs/pythia_capability.csv --out docs/forecast_v2_report.md
  run pythia_timing      docs/pythia_capability.csv $PY -m scripts.pythia_forecast --csv docs/pythia_capability.csv --out docs/pythia_forecast_report.md
  run inductive_bias     "$TR" $PY -m scripts.fit_capability_scaling --results-dir "$TR" --out docs/capability_scaling_report.md
  run forecast_protocol  "$TR" $PY -m scripts.forecast_protocol  --results-dir "$TR" --out docs/forecast_protocol_report.md
fi
if [[ "$PAPER" == *A* ]]; then
  echo "-- Paper A --"
  run where_layers       "$TR" $PY -m scripts.plot_where_layers  --results-dir "$TR" --out docs/figures/where_layers
  run emergence          "$TR" $PY -m scripts.plot_emergence     --results-dir "$TR"
  run scaling_laws       "$NANOPROT_BASE_DIR/sweep_results" $PY -m scripts.scaling_laws
fi

echo "== done: $ok regenerated, $skip skipped (missing data), $fail failed =="
[ "$fail" -eq 0 ]
