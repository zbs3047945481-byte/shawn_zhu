#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/autodl_remote_common.sh"

require_autodl_config

PREFIX="${1:-autodl_hard_warmup_ema}"

ssh_autodl "bash -lc '
  cd \"${AUTODL_PROJECT_DIR}\"
  if [[ -f \"runs/${PREFIX}/status.json\" ]]; then
    echo \"== STATUS ==\"
    cat \"runs/${PREFIX}/status.json\"
  fi
  if [[ -f \"outputs/${PREFIX}/summary.csv\" ]]; then
    echo
    echo \"== SUMMARY ==\"
    cat \"outputs/${PREFIX}/summary.csv\"
  fi
'"
