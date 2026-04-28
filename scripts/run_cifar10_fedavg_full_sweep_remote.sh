#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PREFIX="cifar10_ep1_sweep_$(date -u +%Y%m%dT%H%M%SZ)"
ROUND_NUM=100
POLL_SECONDS=300
BATCH_SIZE=512
NUM_CLIENTS=10
C_FRACTION=0.2
LOCAL_EPOCH=1
LR=0.001
MIN_ROUNDS=40
PATIENCE=20
MIN_DELTA=0.002

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
    --num-clients)
      NUM_CLIENTS="$2"
      shift 2
      ;;
    --c-fraction)
      C_FRACTION="$2"
      shift 2
      ;;
    --lr)
      LR="$2"
      shift 2
      ;;
    --early-stop-min-rounds)
      MIN_ROUNDS="$2"
      shift 2
      ;;
    --early-stop-patience)
      PATIENCE="$2"
      shift 2
      ;;
    --early-stop-min-delta)
      MIN_DELTA="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

run_one() {
  local method="$1"
  local alpha="$2"
  local plugin_name="$3"
  local run_id="${PREFIX}_${method}_a${alpha//./}"

  echo "SUBMIT ${run_id}"
  "${REPO_ROOT}/scripts/run_remote_train.sh" --run-id "${run_id}" -- \
    --round_num "${ROUND_NUM}" \
    --num_of_clients "${NUM_CLIENTS}" \
    --c_fraction "${C_FRACTION}" \
    --local_epoch "${LOCAL_EPOCH}" \
    --batch_size "${BATCH_SIZE}" \
    --dataloader_num_workers 0 \
    --dataloader_pin_memory true \
    --torch_cudnn_benchmark true \
    --gpu true \
    --dataset_name cifar10 \
    --partition_strategy dirichlet \
    --dirichlet_alpha "${alpha}" \
    --min_samples_per_client 1 \
    --enable_quantity_skew true \
    --enable_feature_skew true \
    --lr "${LR}" \
    --early_stop_enable true \
    --early_stop_min_rounds "${MIN_ROUNDS}" \
    --early_stop_patience "${PATIENCE}" \
    --early_stop_min_delta "${MIN_DELTA}" \
    --plugin_name "${plugin_name}"

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

for alpha in 0.1 0.3 0.5; do
  run_one "fedavg" "${alpha}" "none"
  run_one "full" "${alpha}" "fedfed_prototype"
done

echo "SWEEP_DONE prefix=${PREFIX}"
