#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PREFIX="cifar10_a03_ablation_$(date -u +%Y%m%dT%H%M%SZ)"
ROUND_NUM=100
POLL_SECONDS=600
BATCH_SIZE=512

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)
      PREFIX="$2"
      shift 2
      ;;
    --round-num)
      ROUND_NUM="$2"
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

COMMON_ARGS=(
  --round_num "${ROUND_NUM}"
  --num_of_clients 10
  --c_fraction 0.2
  --local_epoch 1
  --batch_size "${BATCH_SIZE}"
  --dataloader_num_workers 0
  --dataloader_pin_memory true
  --torch_cudnn_benchmark true
  --gpu true
  --dataset_name cifar10
  --partition_strategy dirichlet
  --dirichlet_alpha 0.3
  --min_samples_per_client 1
  --enable_quantity_skew true
  --enable_feature_skew true
  --lr 0.001
  --early_stop_enable true
  --early_stop_min_rounds 40
  --early_stop_patience 20
  --early_stop_min_delta 0.002
  --plugin_name fedfed_prototype
)

run_one() {
  local name="$1"
  shift
  local run_id="${PREFIX}_${name}"

  echo "SUBMIT ${run_id}"
  "${REPO_ROOT}/scripts/run_remote_train.sh" --run-id "${run_id}" -- "${COMMON_ARGS[@]}" "$@"

  while true; do
    local summary
    summary="$("${REPO_ROOT}/scripts/check_remote_run_compact.sh" --run-id "${run_id}" --stderr-lines 2)"
    echo "${summary}"
    if grep -q "status=succeeded" <<<"${summary}"; then
      break
    fi
    if grep -q "status=failed" <<<"${summary}"; then
      echo "FAILED ${run_id}" >&2
      return 1
    fi
    sleep "${POLL_SECONDS}"
  done
}

run_one "lambda01" --fedfed_lambda_distill 0.1
run_one "lambda03" --fedfed_lambda_distill 0.3
run_one "warmup10" --fedfed_distill_warmup_rounds 10
run_one "warmup20" --fedfed_distill_warmup_rounds 20
run_one "no_anchor" --fedfed_enable_anchor false
run_one "no_reliability" --fedfed_distill_count_tau 0
run_one "no_ema" --fedfed_prototype_momentum 0
run_one "no_feature_skew" --enable_feature_skew false

echo "ABLATION_DONE prefix=${PREFIX}"
