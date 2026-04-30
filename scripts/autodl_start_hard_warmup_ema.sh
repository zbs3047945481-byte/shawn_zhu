#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/autodl_remote_common.sh"

require_autodl_config

PREFIX="autodl_hard_warmup_ema_$(date -u +%Y%m%dT%H%M%SZ)"
ROUND_NUM=100
BATCH_SIZE=256

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

ssh_autodl "mkdir -p '${AUTODL_PROJECT_DIR}'"
rsync_to_autodl "${REPO_ROOT}/" "${AUTODL_PROJECT_DIR}/"

ssh_autodl "bash -lc '
  set -euo pipefail
  cd \"${AUTODL_PROJECT_DIR}\"
  mkdir -p \"runs/${PREFIX}\"
  nohup ${AUTODL_PYTHON} -u scripts/autodl_hard_warmup_ema_queue.py \
    --prefix \"${PREFIX}\" \
    --round-num \"${ROUND_NUM}\" \
    --batch-size \"${BATCH_SIZE}\" \
    > \"runs/${PREFIX}/queue.log\" 2>&1 &
  echo \"AUTODL_RUN_STARTED prefix=${PREFIX} pid=\$!\"
'"

echo "Started AutoDL hard-warmup EMA queue: ${PREFIX}"
echo "Remote status: ${AUTODL_PROJECT_DIR}/runs/${PREFIX}/status.json"
echo "Remote summary: ${AUTODL_PROJECT_DIR}/outputs/${PREFIX}/summary.csv"
