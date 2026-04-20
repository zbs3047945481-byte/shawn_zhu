#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REMOTE_ENV_FILE="${REMOTE_ENV_FILE:-${SCRIPT_DIR}/remote_windows.env}"

if [[ -f "${REMOTE_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${REMOTE_ENV_FILE}"
fi

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required variable: ${name}" >&2
    exit 1
  fi
}

require_remote_config() {
  require_var WIN_HOST_ALIAS
  require_var REMOTE_PROJECT_DIR
  require_var REMOTE_RUNS_DIR
  require_var REMOTE_PYTHON
  require_var REMOTE_Codex_DIR
}

ssh_target() {
  printf '%s\n' "${WIN_HOST_ALIAS}"
}

ssh_remote() {
  ssh -o BatchMode=yes "$(ssh_target)" "$@"
}

scp_to_remote() {
  local source_path="$1"
  local target_path="$2"
  scp -o BatchMode=yes "${source_path}" "$(ssh_target):${target_path}"
}

timestamp_utc() {
  date -u +"%Y%m%dT%H%M%SZ"
}

make_temp_dir() {
  mktemp -d "${TMPDIR:-/tmp}/codex-remote.XXXXXX"
}

cleanup_temp_dir() {
  local dir_path="$1"
  if [[ -n "${dir_path}" && -d "${dir_path}" ]]; then
    rm -rf "${dir_path}"
  fi
}
