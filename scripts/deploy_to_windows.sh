#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/remote_common.sh"

require_remote_config

TMP_DIR="$(make_temp_dir)"
trap 'cleanup_temp_dir "${TMP_DIR}"' EXIT

LOCAL_MANIFEST_PATH="${TMP_DIR}/local_manifest.tsv"
REMOTE_MANIFEST_CACHE="${TMP_DIR}/remote_manifest.tsv"
CHANGED_LIST_PATH="${TMP_DIR}/changed_files.txt"
DELETED_LIST_PATH="${TMP_DIR}/deleted_files.txt"
FILE_LIST_PATH="${TMP_DIR}/filelist.txt"
PAYLOAD_ROOT="${TMP_DIR}/payload"
DEPLOY_SCRIPT_NAME="deploy_$(timestamp_utc).ps1"
DEPLOY_SCRIPT_PATH="${TMP_DIR}/${DEPLOY_SCRIPT_NAME}"
REMOTE_MANIFEST_PATH="${REMOTE_Codex_DIR}/deploy_manifest.tsv"
REMOTE_MANIFEST_PS_PATH="${REMOTE_MANIFEST_PATH//\//\\}"
REMOTE_DELETED_PATH="${REMOTE_Codex_DIR}/deleted_files.txt"
REMOTE_DELETED_PS_PATH="${REMOTE_DELETED_PATH//\//\\}"
REMOTE_PAYLOAD_DIR="${REMOTE_Codex_DIR}/incremental_payload"
REMOTE_PAYLOAD_PS_DIR="${REMOTE_PAYLOAD_DIR//\//\\}"
REMOTE_DEPLOY_SCRIPT_PATH="${REMOTE_Codex_DIR}/${DEPLOY_SCRIPT_NAME}"
REMOTE_DEPLOY_SCRIPT_PS_PATH="${REMOTE_DEPLOY_SCRIPT_PATH//\//\\}"

(
  cd "${REPO_ROOT}"
  {
    git -c core.quotepath=off ls-files
    git -c core.quotepath=off ls-files --others --exclude-standard
  } | awk 'NF' | LC_ALL=C sort -u > "${FILE_LIST_PATH}"

  if [[ ! -s "${FILE_LIST_PATH}" ]]; then
    echo "No files selected for deployment." >&2
    exit 1
  fi

  : > "${LOCAL_MANIFEST_PATH}"
  while IFS= read -r relpath; do
    if [[ ! -f "${relpath}" ]]; then
      continue
    fi
    hash_value="$(shasum -a 256 "${relpath}" | awk '{print $1}')"
    printf '%s\t%s\n' "${relpath}" "${hash_value}" >> "${LOCAL_MANIFEST_PATH}"
  done < "${FILE_LIST_PATH}"
)

if ! scp -o BatchMode=yes "$(ssh_target):${REMOTE_MANIFEST_PATH}" "${REMOTE_MANIFEST_CACHE}" >/dev/null 2>&1; then
  : > "${REMOTE_MANIFEST_CACHE}"
fi

/usr/bin/python3 - "${LOCAL_MANIFEST_PATH}" "${REMOTE_MANIFEST_CACHE}" "${CHANGED_LIST_PATH}" "${DELETED_LIST_PATH}" <<'PY'
from pathlib import Path
import sys

local_manifest = Path(sys.argv[1])
remote_manifest = Path(sys.argv[2])
changed_out = Path(sys.argv[3])
deleted_out = Path(sys.argv[4])

def load_manifest(path: Path):
    entries = {}
    if not path.exists():
        return entries
    raw = path.read_bytes()
    text = None
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "gbk", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="ignore")
    for line in text.splitlines():
        if not line.strip() or "\t" not in line:
            continue
        relpath, digest = line.split("\t", 1)
        entries[relpath] = digest
    return entries

local = load_manifest(local_manifest)
remote = load_manifest(remote_manifest)

changed = sorted([path for path, digest in local.items() if remote.get(path) != digest])
deleted = sorted([path for path in remote if path not in local])

changed_out.write_text("\n".join(changed) + ("\n" if changed else ""), encoding="utf-8")
deleted_out.write_text("\n".join(deleted) + ("\n" if deleted else ""), encoding="utf-8")
PY

changed_count="$(wc -l < "${CHANGED_LIST_PATH}" | tr -d ' ')"
deleted_count="$(wc -l < "${DELETED_LIST_PATH}" | tr -d ' ')"

echo "Incremental deploy plan: ${changed_count} changed, ${deleted_count} deleted"

ssh_remote "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force -Path '${REMOTE_Codex_DIR}' | Out-Null; New-Item -ItemType Directory -Force -Path '${REMOTE_PROJECT_DIR}' | Out-Null; if (Test-Path '${REMOTE_PAYLOAD_PS_DIR}') { Remove-Item -Recurse -Force '${REMOTE_PAYLOAD_PS_DIR}' }; New-Item -ItemType Directory -Force -Path '${REMOTE_PAYLOAD_PS_DIR}' | Out-Null\""

if [[ "${changed_count}" != "0" ]]; then
  mkdir -p "${PAYLOAD_ROOT}"
  (
    cd "${REPO_ROOT}"
    while IFS= read -r relpath; do
      [[ -z "${relpath}" ]] && continue
      mkdir -p "${PAYLOAD_ROOT}/$(dirname "${relpath}")"
      cp -p "${relpath}" "${PAYLOAD_ROOT}/${relpath}"
    done < "${CHANGED_LIST_PATH}"
  )
  scp -r -o BatchMode=yes "${PAYLOAD_ROOT}/." "$(ssh_target):${REMOTE_PAYLOAD_DIR}"
fi

scp_to_remote "${LOCAL_MANIFEST_PATH}" "${REMOTE_MANIFEST_PATH}"
scp_to_remote "${DELETED_LIST_PATH}" "${REMOTE_DELETED_PATH}"

cat > "${DEPLOY_SCRIPT_PATH}" <<EOF
\$ErrorActionPreference = 'Stop'
\$projectPath = '${REMOTE_PROJECT_DIR}'
\$payloadPath = '${REMOTE_PAYLOAD_PS_DIR}'
\$deletedListPath = '${REMOTE_DELETED_PS_PATH}'
\$changedCount = ${changed_count}

if (-not (Test-Path \$projectPath)) {
    New-Item -ItemType Directory -Force -Path \$projectPath | Out-Null
}

if (\$changedCount -gt 0 -and (Test-Path \$payloadPath)) {
    Copy-Item -Path (Join-Path \$payloadPath '*') -Destination \$projectPath -Recurse -Force
    Remove-Item -Recurse -Force \$payloadPath
}

if (Test-Path \$deletedListPath) {
    Get-Content \$deletedListPath | ForEach-Object {
        if ([string]::IsNullOrWhiteSpace(\$_)) {
            return
        }
        \$target = Join-Path \$projectPath \$_
        if (Test-Path \$target) {
            Remove-Item -Recurse -Force \$target
        }
    }
}

\$deletedCount = 0
if (Test-Path \$deletedListPath) {
    \$deletedCount = (Get-Content \$deletedListPath | Where-Object { -not [string]::IsNullOrWhiteSpace(\$_) }).Count
}
Write-Output ("DEPLOY_OK {0} CHANGED=${changed_count} DELETED={1}" -f \$projectPath, \$deletedCount)
EOF

scp_to_remote "${DEPLOY_SCRIPT_PATH}" "${REMOTE_DEPLOY_SCRIPT_PATH}"
ssh_remote "powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"${REMOTE_DEPLOY_SCRIPT_PS_PATH}\""

echo "Deployment finished."
