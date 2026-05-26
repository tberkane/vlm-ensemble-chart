#!/bin/bash
# Run sensitivity ablations on WB v2 Scout with cached samples.
# Baseline for coverage/tolerance ablations: coverage=0.95, tolerance=0.01, exact_match=false
# Baseline for pruning ablations: coverage=0.95, tolerance=0.01, exact_match=true (matches main Table 1 setup)

set -e
cd "$(dirname "$0")/.."

COMMON_ARGS=(
  --cfg.input_images_dir "data/World Bank v2/test/png"
  --cfg.gt_dir "data/World Bank v2/test/tables"
  --cfg.strategy incremental_ensemble
  --cfg.model "meta-llama/llama-4-scout-17b-16e-instruct"
  --cfg.temperature 0.0
  --cfg.incremental_initial_batch_size 3
  --cfg.incremental_max_samples 20
  --cfg.incremental_patience 2
  --cfg.aggregation median
  --cfg.pruning true
)

run_one() {
  local name=$1
  shift
  local run_dir="outputs/adaptive/wb_v2_scout_${name}"
  if [ -f "${run_dir}/iteration_metrics.json" ]; then
    echo "=== SKIP ${name} (already complete) ==="
    return
  fi
  echo "=== RUN ${name} → ${run_dir} ==="
  rm -rf "${run_dir}"
  python scripts/adaptive/predict.py \
    "${COMMON_ARGS[@]}" \
    "$@" \
    --cfg.run_dir "${run_dir}"
}

# --- Coverage sweep (exact_match=false required for coverage to matter) ---
# Also runs the "default" (0.95) with exact_match=false so the default row matches sensitivity table setup
run_one "em0_cov0p90_tol0p01"   --cfg.incremental_exact_match false --cfg.incremental_convergence_coverage 0.90  --cfg.incremental_convergence_tolerance 0.01 --cfg.pruning_threshold 0.5
run_one "em0_cov0p95_tol0p01"   --cfg.incremental_exact_match false --cfg.incremental_convergence_coverage 0.95  --cfg.incremental_convergence_tolerance 0.01 --cfg.pruning_threshold 0.5
run_one "em0_cov0p975_tol0p01"  --cfg.incremental_exact_match false --cfg.incremental_convergence_coverage 0.975 --cfg.incremental_convergence_tolerance 0.01 --cfg.pruning_threshold 0.5

# --- Tolerance sweep ---
run_one "em0_cov0p95_tol0p001"  --cfg.incremental_exact_match false --cfg.incremental_convergence_coverage 0.95  --cfg.incremental_convergence_tolerance 0.001 --cfg.pruning_threshold 0.5
run_one "em0_cov0p95_tol0p10"   --cfg.incremental_exact_match false --cfg.incremental_convergence_coverage 0.95  --cfg.incremental_convergence_tolerance 0.10  --cfg.pruning_threshold 0.5

# --- Pruning sweep (keep exact_match=true to match main Table 1 configuration) ---
run_one "em1_prune0p0" --cfg.incremental_exact_match true --cfg.incremental_convergence_coverage 0.95 --cfg.incremental_convergence_tolerance 0.01 --cfg.pruning_threshold 0.0
run_one "em1_prune0p1" --cfg.incremental_exact_match true --cfg.incremental_convergence_coverage 0.95 --cfg.incremental_convergence_tolerance 0.01 --cfg.pruning_threshold 0.1
run_one "em1_prune0p2" --cfg.incremental_exact_match true --cfg.incremental_convergence_coverage 0.95 --cfg.incremental_convergence_tolerance 0.01 --cfg.pruning_threshold 0.2
run_one "em1_prune0p3" --cfg.incremental_exact_match true --cfg.incremental_convergence_coverage 0.95 --cfg.incremental_convergence_tolerance 0.01 --cfg.pruning_threshold 0.3
run_one "em1_prune0p4" --cfg.incremental_exact_match true --cfg.incremental_convergence_coverage 0.95 --cfg.incremental_convergence_tolerance 0.01 --cfg.pruning_threshold 0.4

echo "=== ALL ABLATIONS COMPLETE ==="
