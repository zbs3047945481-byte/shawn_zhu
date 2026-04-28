#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/remote_common.sh"

require_remote_config

RUN_ID=""
STDERR_LINES=3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --stderr-lines)
      STDERR_LINES="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 --run-id RUN_ID [--stderr-lines N]" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${RUN_ID}" ]]; then
  echo "Missing --run-id" >&2
  exit 1
fi

TMP_DIR="$(make_temp_dir)"
trap 'cleanup_temp_dir "${TMP_DIR}"' EXIT
CHECKER_FILE="${TMP_DIR}/compact_checker.ps1"
REMOTE_CHECKER_PATH="${REMOTE_Codex_DIR}/compact_checker_${RUN_ID}.ps1"

cat > "${CHECKER_FILE}" <<EOF
\$ErrorActionPreference = 'Stop'
\$runId = '${RUN_ID}'
\$runsRoot = '${REMOTE_RUNS_DIR}'
\$projectPath = '${REMOTE_PROJECT_DIR}'
\$stderrLines = ${STDERR_LINES}
\$runDir = Join-Path \$runsRoot \$runId
\$statusPath = Join-Path \$runDir 'run_status.json'
\$metaPath = Join-Path \$runDir 'run_meta.json'
\$stderrPath = Join-Path \$runDir 'stderr.log'

if (!(Test-Path \$runDir)) {
  Write-Output ("RUN {0} status=missing" -f \$runId)
  exit 0
}

\$status = \$null
if (Test-Path \$statusPath) {
  \$status = Get-Content \$statusPath -Raw | ConvertFrom-Json
}
\$meta = \$null
if (Test-Path \$metaPath) {
  \$meta = Get-Content \$metaPath -Raw | ConvertFrom-Json
}
\$pidText = ''
if (\$status -and \$status.pid) {
  \$pidText = [string]\$status.pid
} elseif (\$meta -and \$meta.pid) {
  \$pidText = [string]\$meta.pid
}
\$procState = 'unknown'
if (\$pidText -ne '') {
  \$proc = Get-Process -Id ([int]\$pidText) -ErrorAction SilentlyContinue
  \$procState = if (\$proc) { 'running' } else { 'not_running' }
}
\$statusText = if (\$status) { \$status.status } else { 'unknown' }
Write-Output ("RUN {0} status={1} pid={2} proc={3}" -f \$runId, \$statusText, \$pidText, \$procState)

\$metricsRoot = Join-Path \$projectPath 'result'
\$metricsFiles = @()
if (Test-Path \$metricsRoot) {
  \$metricsFiles = Get-ChildItem -Path \$metricsRoot -Recurse -Include metrics.json,live_metrics.json -File |
    Where-Object { \$_.DirectoryName -like ('*' + \$runId + '*') } |
    Sort-Object LastWriteTime -Descending
}
if (\$metricsFiles.Count -gt 0) {
  \$metricsFile = \$metricsFiles | Where-Object { \$_.Name -eq 'metrics.json' } | Select-Object -First 1
  if (\$null -eq \$metricsFile) {
    \$metricsFile = \$metricsFiles | Select-Object -First 1
  }
  \$m = Get-Content \$metricsFile.FullName -Raw | ConvertFrom-Json
  \$round = ''
  if (\$null -ne \$m.final_round) {
    \$round = \$m.final_round
  } elseif (\$m.rounds.Count -gt 0) {
    \$round = \$m.rounds[-1]
  }
  \$finalAcc = if (\$null -ne \$m.final_test_acc) { [double]\$m.final_test_acc } else { [double]0 }
  \$bestAcc = if (\$null -ne \$m.best_test_acc) { [double]\$m.best_test_acc } else { [double]0 }
  \$finalLoss = if (\$null -ne \$m.final_test_loss) { [double]\$m.final_test_loss } else { [double]0 }
  Write-Output ("METRICS file={0} round={1} final_acc={2:F2}% best_acc={3:F2}% final_loss={4:F4}" -f \$metricsFile.Name, \$round, (100*\$finalAcc), (100*\$bestAcc), \$finalLoss)
} else {
  Write-Output 'METRICS none'
}

\$gpuLine = (& nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>\$null | Select-Object -First 1)
if (\$LASTEXITCODE -eq 0 -and \$gpuLine) {
  \$parts = \$gpuLine.Split(',') | ForEach-Object { \$_.Trim() }
  Write-Output ("GPU util={0}% mem={1}/{2}MiB" -f \$parts[0], \$parts[1], \$parts[2])
}

if ((Test-Path \$stderrPath) -and ((Get-Item \$stderrPath).Length -gt 0) -and (\$stderrLines -gt 0)) {
  Write-Output 'STDERR_TAIL_BEGIN'
  Get-Content \$stderrPath -Tail \$stderrLines
  Write-Output 'STDERR_TAIL_END'
}
EOF

ssh_remote "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force -Path '${REMOTE_Codex_DIR}' | Out-Null\""
scp_to_remote "${CHECKER_FILE}" "${REMOTE_CHECKER_PATH}"
ssh_remote "powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"${REMOTE_CHECKER_PATH}\""
