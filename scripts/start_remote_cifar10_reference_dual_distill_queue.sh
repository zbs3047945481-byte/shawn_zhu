#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/remote_common.sh"

require_remote_config

PREFIX="cifar10_reference_dual_distill_$(date -u +%Y%m%dT%H%M%SZ)"
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

TMP_DIR="$(make_temp_dir)"
trap 'cleanup_temp_dir "${TMP_DIR}"' EXIT

QUEUE_FILE="${TMP_DIR}/reference_dual_distill_queue.ps1"
LAUNCHER_FILE="${TMP_DIR}/reference_dual_distill_queue_launcher.ps1"
REMOTE_QUEUE_PATH="${REMOTE_Codex_DIR}/reference_dual_distill_queue_${PREFIX}.ps1"
REMOTE_LAUNCHER_PATH="${REMOTE_Codex_DIR}/reference_dual_distill_queue_launcher_${PREFIX}.ps1"
REMOTE_QUEUE_PS_PATH="${REMOTE_QUEUE_PATH//\//\\}"

cat > "${QUEUE_FILE}" <<EOF
\$ErrorActionPreference = 'Stop'
\$pythonPath = '${REMOTE_PYTHON}'
\$projectPath = '${REMOTE_PROJECT_DIR}'
\$runsRoot = '${REMOTE_RUNS_DIR}'
\$prefix = '${PREFIX}'
\$queueStatusPath = Join-Path \$runsRoot ("\$prefix" + '_queue_status.json')

function Write-QueueStatus(\$status, \$currentRun, \$message) {
  [ordered]@{
    prefix = \$prefix
    status = \$status
    current_run = \$currentRun
    message = \$message
    updated_utc = (Get-Date).ToUniversalTime().ToString('o')
  } | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 \$queueStatusPath
}

function Run-One(\$name, \$pluginName, \$extraArgs) {
  \$runId = "\${prefix}_\${name}_a03_ep5"
  \$runDir = Join-Path \$runsRoot \$runId
  \$stdoutPath = Join-Path \$runDir 'stdout.log'
  \$stderrPath = Join-Path \$runDir 'stderr.log'
  \$metaPath = Join-Path \$runDir 'run_meta.json'
  \$statusPath = Join-Path \$runDir 'run_status.json'
  New-Item -ItemType Directory -Force -Path \$runDir | Out-Null
  New-Item -ItemType File -Force -Path \$stdoutPath | Out-Null
  New-Item -ItemType File -Force -Path \$stderrPath | Out-Null

  \$trainArgs = @(
    '--round_num', '${ROUND_NUM}',
    '--num_of_clients', '20',
    '--c_fraction', '0.2',
    '--local_epoch', '5',
    '--batch_size', '${BATCH_SIZE}',
    '--dataloader_num_workers', '0',
    '--dataloader_pin_memory', 'true',
    '--torch_cudnn_benchmark', 'true',
    '--gpu', 'true',
    '--dataset_name', 'cifar10',
    '--partition_strategy', 'dirichlet',
    '--dirichlet_alpha', '0.3',
    '--min_samples_per_client', '1',
    '--enable_quantity_skew', 'true',
    '--enable_feature_skew', 'true',
    '--lr', '0.001',
    '--early_stop_enable', 'true',
    '--early_stop_min_rounds', '40',
    '--early_stop_patience', '20',
    '--early_stop_min_delta', '0.002',
    '--plugin_name', \$pluginName,
    '--experiment_tag', \$runId
  ) + \$extraArgs

  \$startedUtc = (Get-Date).ToUniversalTime().ToString('o')
  [ordered]@{
    run_id = \$runId
    pid = \$PID
    project_path = \$projectPath
    python_path = \$pythonPath
    stdout_log = \$stdoutPath
    stderr_log = \$stderrPath
    launched_utc = \$startedUtc
    mode = 'reference_dual_distill_queue'
    arguments = \$trainArgs
  } | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 \$metaPath

  [ordered]@{
    run_id = \$runId
    status = 'running'
    pid = \$PID
    started_utc = \$startedUtc
    exit_code = \$null
    finished_utc = \$null
  } | ConvertTo-Json -Depth 3 | Set-Content -Encoding UTF8 \$statusPath

  Write-QueueStatus 'running' \$runId 'Training.'
  Set-Location \$projectPath
  \$ErrorActionPreference = 'Continue'
  & \$pythonPath '-u' 'main.py' @trainArgs 1>> \$stdoutPath 2>> \$stderrPath
  \$exitCode = if (\$LASTEXITCODE -ne \$null) { \$LASTEXITCODE } else { 0 }
  \$ErrorActionPreference = 'Stop'
  \$finalStatus = if (\$exitCode -eq 0) { 'succeeded' } else { 'failed' }

  [ordered]@{
    run_id = \$runId
    status = \$finalStatus
    pid = \$PID
    started_utc = \$startedUtc
    exit_code = \$exitCode
    finished_utc = (Get-Date).ToUniversalTime().ToString('o')
  } | ConvertTo-Json -Depth 3 | Set-Content -Encoding UTF8 \$statusPath

  if (\$exitCode -ne 0) {
    Write-QueueStatus 'failed' \$runId "Training failed with exit code \$exitCode."
    throw "Run failed: \$runId"
  }
}

\$baseDualArgs = @(
  '--fedfed_enable_projection', 'false',
  '--fedfed_sensitive_dim', '512',
  '--fedfed_distill_count_tau', '0',
  '--fedfed_normalize_prototypes', 'false',
  '--fedfed_use_cosine_distill', 'false',
  '--fedfed_prototype_momentum', '0.8',
  '--fedfed_prototype_source', 'reference',
  '--fedfed_reference_proto_max_batches', '0',
  '--fedfed_enable_proto_cls', 'false',
  '--fedfed_enable_clip', 'false',
  '--fedfed_enable_noise', 'false',
  '--fedfed_adaptive_control', 'false',
  '--fedfed_enable_anchor', 'true',
  '--fedfed_lambda_anchor', '0.1',
  '--fedfed_enable_distill', 'true'
)

try {
  New-Item -ItemType Directory -Force -Path \$runsRoot | Out-Null
  Write-QueueStatus 'starting' '' 'CIFAR-10 reference-prototype dual-distill grid starting.'
  Run-One 'fedavg' 'none' @()
  Run-One 'lambda01_warmup0' 'fedfed_prototype' (\$baseDualArgs + @('--fedfed_lambda_distill', '0.1', '--fedfed_distill_warmup_rounds', '0'))
  Run-One 'lambda01_warmup5' 'fedfed_prototype' (\$baseDualArgs + @('--fedfed_lambda_distill', '0.1', '--fedfed_distill_warmup_rounds', '5'))
  Run-One 'lambda01_warmup10' 'fedfed_prototype' (\$baseDualArgs + @('--fedfed_lambda_distill', '0.1', '--fedfed_distill_warmup_rounds', '10'))
  Run-One 'lambda03_warmup0' 'fedfed_prototype' (\$baseDualArgs + @('--fedfed_lambda_distill', '0.3', '--fedfed_distill_warmup_rounds', '0'))
  Run-One 'lambda03_warmup5' 'fedfed_prototype' (\$baseDualArgs + @('--fedfed_lambda_distill', '0.3', '--fedfed_distill_warmup_rounds', '5'))
  Run-One 'lambda03_warmup10' 'fedfed_prototype' (\$baseDualArgs + @('--fedfed_lambda_distill', '0.3', '--fedfed_distill_warmup_rounds', '10'))
  Write-QueueStatus 'succeeded' '' 'CIFAR-10 reference-prototype dual-distill grid finished.'
} catch {
  Add-Content -Path (Join-Path \$runsRoot ("\$prefix" + '_queue_error.log')) -Value (\$_ | Out-String)
  Write-QueueStatus 'failed' '' (\$_ | Out-String)
  exit 1
}
EOF

cat > "${LAUNCHER_FILE}" <<EOF
\$ErrorActionPreference = 'Stop'
\$queuePath = '${REMOTE_QUEUE_PS_PATH}'
\$commandLine = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "' + \$queuePath + '"'
\$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = \$commandLine }
if (\$result.ReturnValue -ne 0) {
  throw ("WIN32_PROCESS_CREATE_FAILED return_value={0}" -f \$result.ReturnValue)
}
Write-Output ("QUEUE_OK ${PREFIX} PID={0}" -f \$result.ProcessId)
EOF

ssh_remote "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force -Path '${REMOTE_Codex_DIR}' | Out-Null\""
scp_to_remote "${QUEUE_FILE}" "${REMOTE_QUEUE_PATH}"
scp_to_remote "${LAUNCHER_FILE}" "${REMOTE_LAUNCHER_PATH}"
ssh_remote "powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"${REMOTE_LAUNCHER_PATH}\""
echo "Offline reference dual-distill queue submitted: ${PREFIX}"
