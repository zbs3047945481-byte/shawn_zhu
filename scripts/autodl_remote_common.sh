#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AUTODL_ENV_FILE="${AUTODL_ENV_FILE:-${SCRIPT_DIR}/autodl_remote.env}"

if [[ -f "${AUTODL_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${AUTODL_ENV_FILE}"
fi

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required variable: ${name}" >&2
    exit 1
  fi
}

require_autodl_config() {
  require_var AUTODL_HOST
  require_var AUTODL_PORT
  require_var AUTODL_USER
  require_var AUTODL_PROJECT_DIR
  AUTODL_PYTHON="${AUTODL_PYTHON:-python}"
}

autodl_target() {
  printf '%s@%s\n' "${AUTODL_USER}" "${AUTODL_HOST}"
}

ssh_autodl() {
  ssh -p "${AUTODL_PORT}" "$(autodl_target)" "$@"
}

rsync_to_autodl() {
  local source_path="$1"
  local target_path="$2"
  rsync -az --delete \
    --exclude '.git' \
    --exclude '.codex' \
    --exclude 'data' \
    --exclude 'result' \
    --exclude 'runs' \
    --exclude 'outputs' \
    --exclude '.npm-cache' \
    -e "ssh -p ${AUTODL_PORT}" \
    "${source_path}" "$(autodl_target):${target_path}"
}
