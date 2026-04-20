#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/remote_common.sh"

require_remote_config

RUN_ID=""
TAIL_LINES="${TAIL_LINES:-60}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --tail-lines)
      TAIL_LINES="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--run-id RUN_ID] [--tail-lines N]" >&2
      exit 1
      ;;
  esac
done

RUN_ID_ESCAPED="${RUN_ID//\'/\'\'}"
TMP_DIR="$(make_temp_dir)"
trap 'cleanup_temp_dir "${TMP_DIR}"' EXIT
CHECK_SCRIPT_NAME="check_$(timestamp_utc).ps1"
CHECK_SCRIPT_PATH="${TMP_DIR}/${CHECK_SCRIPT_NAME}"
REMOTE_CHECK_SCRIPT_PATH="${REMOTE_Codex_DIR}/${CHECK_SCRIPT_NAME}"

cat > "${CHECK_SCRIPT_PATH}" <<EOF
\$ErrorActionPreference = 'Stop'
\$runsRoot = '${REMOTE_RUNS_DIR}'
\$projectPath = '${REMOTE_PROJECT_DIR}'
\$requestedRunId = '${RUN_ID_ESCAPED}'
\$tailLines = ${TAIL_LINES}

if (-not (Test-Path \$runsRoot)) {
    throw "Runs root does not exist: \$runsRoot"
}

if ([string]::IsNullOrWhiteSpace(\$requestedRunId)) {
    \$runDir = Get-ChildItem -Path \$runsRoot -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not \$runDir) {
        throw "No run directories found under \$runsRoot"
    }
} else {
    \$runDir = Get-Item -Path (Join-Path \$runsRoot \$requestedRunId)
}

\$metaPath = Join-Path \$runDir.FullName 'run_meta.json'
\$statusPath = Join-Path \$runDir.FullName 'run_status.json'
\$stdoutPath = Join-Path \$runDir.FullName 'stdout.log'
\$stderrPath = Join-Path \$runDir.FullName 'stderr.log'

Write-Output '== RUN DIRECTORY =='
Write-Output \$runDir.FullName

if (Test-Path \$metaPath) {
    Write-Output '== RUN META =='
    Get-Content \$metaPath
    \$meta = Get-Content \$metaPath | ConvertFrom-Json
    \$proc = Get-Process -Id \$meta.pid -ErrorAction SilentlyContinue
    Write-Output '== PROCESS STATUS =='
    if (\$proc) {
        Write-Output ("PID {0} RUNNING" -f \$proc.Id)
    } else {
        Write-Output ("PID {0} NOT_RUNNING" -f \$meta.pid)
    }
}

if (Test-Path \$statusPath) {
    Write-Output '== RUN STATUS =='
    Get-Content \$statusPath
}

Write-Output '== GPU STATUS =='
try {
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
} catch {
    Write-Output 'nvidia-smi unavailable'
}

Write-Output '== STDOUT TAIL =='
if (Test-Path \$stdoutPath) {
    Get-Content \$stdoutPath -Tail \$tailLines
} else {
    Write-Output 'stdout.log not found'
}

Write-Output '== STDERR TAIL =='
if (Test-Path \$stderrPath) {
    Get-Content \$stderrPath -Tail \$tailLines
} else {
    Write-Output 'stderr.log not found'
}

Write-Output '== LATEST METRICS =='
\$metrics = Get-ChildItem -Path (Join-Path \$projectPath 'result') -Filter metrics.json -Recurse -ErrorAction SilentlyContinue |
  Where-Object { \$_.FullName -like ('*' + \$runDir.Name + '*') } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
if (-not \$metrics) {
    \$metrics = Get-ChildItem -Path (Join-Path \$projectPath 'result') -Filter metrics.json -Recurse -ErrorAction SilentlyContinue |
      Sort-Object LastWriteTime -Descending |
      Select-Object -First 1
}
if (\$metrics) {
    Write-Output \$metrics.FullName
    Get-Content \$metrics.FullName
} else {
    Write-Output 'metrics.json not found under result/'
}
EOF

ssh_remote "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force -Path '${REMOTE_Codex_DIR}' | Out-Null\""
scp_to_remote "${CHECK_SCRIPT_PATH}" "${REMOTE_CHECK_SCRIPT_PATH}"
ssh_remote "powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"${REMOTE_CHECK_SCRIPT_PATH}\""
