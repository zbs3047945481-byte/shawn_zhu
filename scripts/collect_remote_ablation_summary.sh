#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/remote_common.sh"

require_remote_config

PREFIX="cifar10_a03_ablation_20260428"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)
      PREFIX="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--prefix PREFIX]" >&2
      exit 1
      ;;
  esac
done

TMP_DIR="$(make_temp_dir)"
trap 'cleanup_temp_dir "${TMP_DIR}"' EXIT
COLLECTOR_FILE="${TMP_DIR}/collect_ablation_summary.ps1"
REMOTE_COLLECTOR_PATH="${REMOTE_Codex_DIR}/collect_ablation_summary_${PREFIX}.ps1"

cat > "${COLLECTOR_FILE}" <<EOF
\$ErrorActionPreference = 'Stop'
\$prefix = '${PREFIX}'
\$projectPath = '${REMOTE_PROJECT_DIR}'
\$runsRoot = '${REMOTE_RUNS_DIR}'
\$queueStatusPath = Join-Path \$runsRoot ("\$prefix" + '_queue_status.json')

if (Test-Path \$queueStatusPath) {
  Write-Output '== QUEUE =='
  Get-Content \$queueStatusPath -Raw
}

\$root = Join-Path \$projectPath 'result\cifar10'
\$rows = @()
if (Test-Path \$root) {
  \$files = Get-ChildItem -Path \$root -Recurse -Include metrics.json,live_metrics.json -File |
    Where-Object { \$_.DirectoryName -like ('*' + \$prefix + '*') } |
    Sort-Object LastWriteTime -Descending
  \$dirs = \$files | Group-Object DirectoryName
  foreach (\$dir in \$dirs) {
    \$metricsFile = \$dir.Group | Where-Object { \$_.Name -eq 'metrics.json' } | Select-Object -First 1
    if (\$null -eq \$metricsFile) {
      \$metricsFile = \$dir.Group | Select-Object -First 1
    }
    \$m = Get-Content \$metricsFile.FullName -Raw | ConvertFrom-Json
    \$rows += [PSCustomObject]@{
      tag = \$m.options.experiment_tag
      plugin = \$m.plugin_name
      alpha = \$m.options.dirichlet_alpha
      epoch = \$m.options.local_epoch
      adaptive = \$m.options.fedfed_adaptive_control
      round = \$m.final_round
      final = [math]::Round(100 * [double]\$m.final_test_acc, 2)
      best = [math]::Round(100 * [double]\$m.best_test_acc, 2)
      file = \$metricsFile.Name
    }
  }
}

Write-Output '== RESULTS =='
if (\$rows.Count -eq 0) {
  Write-Output 'No metrics found.'
} else {
  \$rows | Sort-Object tag | Format-Table -AutoSize
}
EOF

ssh_remote "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force -Path '${REMOTE_Codex_DIR}' | Out-Null\""
scp_to_remote "${COLLECTOR_FILE}" "${REMOTE_COLLECTOR_PATH}"
ssh_remote "powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"${REMOTE_COLLECTOR_PATH}\""
