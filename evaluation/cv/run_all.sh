#!/usr/bin/env bash
# Full CV evaluation: every model through the same slices, then CPU latency,
# then per-run reports and one comparison table with a Pareto figure.
#
#   bash evaluation/cv/run_all.sh                 # everything
#   SKIP_EXISTING=1 bash evaluation/cv/run_all.sh # keep runs that already have predictions.csv
#
# Prerequisites (see evaluation/cv/README.md): evaluation/cv/data/ populated, masks generated,
# linear-probe exports unpacked under LP_DIR.
set -euo pipefail
cd "$(dirname "$0")/../.."  # repository root (the script lives in evaluation/cv/)

PY="${PY:-.venv/bin/python}"
LP_DIR="${LP_DIR:-evaluation/cv/kaggle_outputs/linear_probe/fishora_linear_probe}"

# run name = export directory
MODELS=(
  "lp_vit_l=$LP_DIR/vit_l_dinov3/export"
  "lp_vit_b=$LP_DIR/vit_b_dinov3/export"
  "lp_vit_s=$LP_DIR/vit_s_dinov3/export"
  "lp_convnext_t=$LP_DIR/convnext_t_dinov3/export"
  "lp_efficientnet_b0=$LP_DIR/efficientnet_b0/export"
  "lp_mobilenetv3_l=$LP_DIR/mobilenetv3_l/export"
)

runs=()
for pair in "${MODELS[@]}"; do
  run="${pair%%=*}"
  export_dir="${pair#*=}"
  runs+=("$run")
  if [[ "${SKIP_EXISTING:-0}" == "1" && -f "evaluation/cv/results/$run/predictions.csv" ]]; then
    echo "== $run: exists, skipping suite"
  else
    echo "== $run: suite"
    "$PY" -u -m evaluation.cv.cv_suite --run "$run" --export "$export_dir"
  fi
  "$PY" -m evaluation.cv.report "$run" > /dev/null
done

echo "== latency (CPU, one subprocess per model)"
"$PY" -u -m evaluation.cv.bench_latency "${MODELS[@]}"

echo "== comparison"
"$PY" -m evaluation.cv.report --compare "${runs[@]}"
