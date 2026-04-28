#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/remote_common.sh"

require_remote_config

RUN_ID=""
PRESET=""
SKIP_DEPLOY=false
TAIL_LINES="${TAIL_LINES:-80}"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_windows_experiment.sh [--run-id RUN_ID] [--preset PRESET] [--skip-deploy] [--tail-lines N] -- <main.py args...>

Presets:
  plugin-smoke   1-round MNIST smoke run with fedfed_prototype on GPU
  fedavg-smoke   1-round MNIST smoke run with plain FedAvg on GPU
EOF
}

preset_args() {
  case "$1" in
    plugin-smoke)
      printf '%s\n' \
        --round_num 1 \
        --num_of_clients 5 \
        --c_fraction 0.4 \
        --local_epoch 1 \
        --batch_size 64 \
        --gpu true \
        --dataset_name mnist \
        --partition_strategy dirichlet \
        --dirichlet_alpha 0.3 \
        --enable_quantity_skew true \
        --enable_feature_skew true \
        --plugin_name fedfed_prototype
      ;;
    fedavg-smoke)
      printf '%s\n' \
        --round_num 1 \
        --num_of_clients 5 \
        --c_fraction 0.4 \
        --local_epoch 1 \
        --batch_size 64 \
        --gpu true \
        --dataset_name mnist \
        --partition_strategy dirichlet \
        --dirichlet_alpha 0.3 \
        --enable_quantity_skew true \
        --enable_feature_skew true \
        --plugin_name none
      ;;
    *)
      echo "Unknown preset: $1" >&2
      exit 1
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --preset)
      PRESET="$2"
      shift 2
      ;;
    --skip-deploy)
      SKIP_DEPLOY=true
      shift
      ;;
    --tail-lines)
      TAIL_LINES="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -n "${PRESET}" && ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "Use either --preset or explicit args after --, not both." >&2
  exit 1
fi

if [[ -n "${PRESET}" ]]; then
  while IFS= read -r line; do
    EXTRA_ARGS+=("${line}")
  done < <(preset_args "${PRESET}")
fi

if [[ ${#EXTRA_ARGS[@]} -eq 0 ]]; then
  echo "No experiment arguments supplied." >&2
  usage >&2
  exit 1
fi

if [[ -z "${RUN_ID}" ]]; then
  if [[ -n "${PRESET}" ]]; then
    RUN_ID="${PRESET//[^a-zA-Z0-9_-]/_}_$(timestamp_utc)"
  else
    RUN_ID="winexp_$(timestamp_utc)"
  fi
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

if [[ "${SKIP_DEPLOY}" == false ]]; then
  "${SCRIPT_DIR}/deploy_to_windows.sh"
fi

ps_array_entries=()
for arg in "${EXTRA_ARGS[@]}"; do
  escaped_arg="${arg//\'/\'\'}"
  ps_array_entries+=("'${escaped_arg}'")
done
PS_ARGS_LITERAL="@($(IFS=', '; echo "${ps_array_entries[*]}"))"

TMP_DIR="$(make_temp_dir)"
trap 'cleanup_temp_dir "${TMP_DIR}"' EXIT
RUNNER_NAME="foreground_${RUN_ID}.ps1"
RUNNER_PATH="${TMP_DIR}/${RUNNER_NAME}"
REMOTE_RUNNER_PATH="${REMOTE_Codex_DIR}/${RUNNER_NAME}"
REMOTE_RUNNER_PS_PATH="${REMOTE_RUNNER_PATH//\//\\}"

cat > "${RUNNER_PATH}" <<EOF
\$ErrorActionPreference = 'Stop'
\$pythonPath = '${REMOTE_PYTHON}'
\$projectPath = '${REMOTE_PROJECT_DIR}'
\$runsRoot = '${REMOTE_RUNS_DIR}'
\$runId = '${RUN_ID}'
\$runDir = Join-Path \$runsRoot \$runId
\$stdoutPath = Join-Path \$runDir 'stdout.log'
\$stderrPath = Join-Path \$runDir 'stderr.log'
\$metaPath = Join-Path \$runDir 'run_meta.json'
\$args = ${PS_ARGS_LITERAL}

New-Item -ItemType Directory -Force -Path \$runDir | Out-Null
New-Item -ItemType File -Force -Path \$stderrPath | Out-Null

[ordered]@{
  run_id = \$runId
  pid = \$PID
  project_path = \$projectPath
  python_path = \$pythonPath
  stdout_log = \$stdoutPath
  stderr_log = \$stderrPath
  launched_utc = (Get-Date).ToUniversalTime().ToString('o')
  mode = 'foreground'
  arguments = \$args
} | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 \$metaPath

Set-Location \$projectPath
& \$pythonPath '-u' 'main.py' @args 2>&1 | Tee-Object -FilePath \$stdoutPath
\$exitCode = \$LASTEXITCODE

if (\$exitCode -ne 0) {
  Write-Error ("TRAIN_FAILED exit_code={0}" -f \$exitCode)
  exit \$exitCode
}
EOF

echo "Submitting foreground experiment ${RUN_ID} to $(ssh_target)..."
ssh_remote "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force -Path '${REMOTE_Codex_DIR}' | Out-Null\""
scp_to_remote "${RUNNER_PATH}" "${REMOTE_RUNNER_PATH}"
ssh_remote "powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"${REMOTE_RUNNER_PS_PATH}\""

echo "Experiment finished: ${RUN_ID}"
"${SCRIPT_DIR}/check_remote_run.sh" --run-id "${RUN_ID}" --tail-lines "${TAIL_LINES}"
