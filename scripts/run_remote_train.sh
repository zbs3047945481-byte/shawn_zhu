#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/remote_common.sh"

require_remote_config

RUN_ID=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--run-id RUN_ID] -- <main.py args...>" >&2
      exit 1
      ;;
  esac
done

if [[ ${#EXTRA_ARGS[@]} -eq 0 ]]; then
  echo "No training arguments supplied." >&2
  echo "Example: $0 -- --round_num 1 --num_of_clients 5 --c_fraction 0.4 --local_epoch 1 --batch_size 64 --gpu true --dataset_name mnist --partition_strategy dirichlet --dirichlet_alpha 0.3 --enable_quantity_skew true --enable_feature_skew true --plugin_name fedfed_prototype" >&2
  exit 1
fi

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="run_$(timestamp_utc)"
fi

has_experiment_tag=false
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  if [[ "${EXTRA_ARGS[$i]}" == "--experiment_tag" ]]; then
    has_experiment_tag=true
    break
  fi
done
if [[ "${has_experiment_tag}" == false ]]; then
  EXTRA_ARGS+=(--experiment_tag "${RUN_ID}")
fi

ps_array_entries=()
for arg in "${EXTRA_ARGS[@]}"; do
  escaped_arg="${arg//\'/\'\'}"
  ps_array_entries+=("'${escaped_arg}'")
done
PS_ARGS_LITERAL="@($(IFS=', '; echo "${ps_array_entries[*]}"))"

TMP_DIR="$(make_temp_dir)"
trap 'cleanup_temp_dir "${TMP_DIR}"' EXIT
WORKER_FILE="${TMP_DIR}/worker.ps1"
LAUNCHER_FILE="${TMP_DIR}/launcher.ps1"
REMOTE_WORKER_PATH="${REMOTE_Codex_DIR}/worker_${RUN_ID}.ps1"
REMOTE_LAUNCHER_PATH="${REMOTE_Codex_DIR}/launcher_${RUN_ID}.ps1"
REMOTE_WORKER_PS_PATH="${REMOTE_WORKER_PATH//\//\\}"

cat > "${WORKER_FILE}" <<EOF
\$ErrorActionPreference = 'Stop'
\$pythonPath = '${REMOTE_PYTHON}'
\$projectPath = '${REMOTE_PROJECT_DIR}'
\$runsRoot = '${REMOTE_RUNS_DIR}'
\$runId = '${RUN_ID}'
\$runDir = Join-Path \$runsRoot \$runId
\$stdoutPath = Join-Path \$runDir 'stdout.log'
\$stderrPath = Join-Path \$runDir 'stderr.log'
\$metaPath = Join-Path \$runDir 'run_meta.json'
\$statusPath = Join-Path \$runDir 'run_status.json'
\$args = ${PS_ARGS_LITERAL}

New-Item -ItemType Directory -Force -Path \$runDir | Out-Null
New-Item -ItemType File -Force -Path \$stdoutPath | Out-Null
New-Item -ItemType File -Force -Path \$stderrPath | Out-Null

\$startedUtc = (Get-Date).ToUniversalTime().ToString('o')

[ordered]@{
  run_id = \$runId
  pid = \$PID
  project_path = \$projectPath
  python_path = \$pythonPath
  stdout_log = \$stdoutPath
  stderr_log = \$stderrPath
  launched_utc = \$startedUtc
  mode = 'background'
  arguments = \$args
} | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 \$metaPath

[ordered]@{
  run_id = \$runId
  status = 'running'
  pid = \$PID
  started_utc = \$startedUtc
  exit_code = \$null
  finished_utc = \$null
} | ConvertTo-Json -Depth 3 | Set-Content -Encoding UTF8 \$statusPath

try {
  Set-Location \$projectPath
  & \$pythonPath '-u' 'main.py' @args 1>> \$stdoutPath 2>> \$stderrPath
  \$exitCode = if (\$LASTEXITCODE -ne \$null) { \$LASTEXITCODE } else { 0 }
} catch {
  \$message = (\$_ | Out-String)
  Add-Content -Path \$stderrPath -Value \$message
  \$exitCode = 1
}

\$finalStatus = if (\$exitCode -eq 0) { 'succeeded' } else { 'failed' }
[ordered]@{
  run_id = \$runId
  status = \$finalStatus
  pid = \$PID
  started_utc = \$startedUtc
  exit_code = \$exitCode
  finished_utc = (Get-Date).ToUniversalTime().ToString('o')
} | ConvertTo-Json -Depth 3 | Set-Content -Encoding UTF8 \$statusPath

exit \$exitCode
EOF

cat > "${LAUNCHER_FILE}" <<EOF
\$ErrorActionPreference = 'Stop'
\$workerPath = '${REMOTE_WORKER_PS_PATH}'
\$runId = '${RUN_ID}'
\$commandLine = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "' + \$workerPath + '"'
\$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = \$commandLine }
if (\$result.ReturnValue -ne 0) {
  throw ("WIN32_PROCESS_CREATE_FAILED return_value={0}" -f \$result.ReturnValue)
}
Write-Output ("RUN_OK {0} PID={1}" -f \$runId, \$result.ProcessId)
EOF

echo "Creating remote run directory ${RUN_ID}..."
ssh_remote "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force -Path '${REMOTE_RUNS_DIR}' | Out-Null\""
ssh_remote "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force -Path '${REMOTE_Codex_DIR}' | Out-Null\""
scp_to_remote "${WORKER_FILE}" "${REMOTE_WORKER_PATH}"
scp_to_remote "${LAUNCHER_FILE}" "${REMOTE_LAUNCHER_PATH}"
ssh_remote "powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"${REMOTE_LAUNCHER_PATH}\""

echo "Run submitted: ${RUN_ID}"
echo "Next step: ./scripts/check_remote_run.sh --run-id ${RUN_ID}"
