#!/usr/bin/env bash
# Download the PRC Data Challenge 2026 datasets from the team's OpenSky bucket.
#
# CREDENTIALS ARE NEVER HANDLED BY THIS REPOSITORY.
# Before running this, configure the MinIO client yourself, once:
#
#   mc alias set dc26 https://s3.opensky-network.org/ <ACCESS_KEY> <SECRET_KEY>
#
# mc stores the alias in ~/.mc/config.json, outside this repository. This script
# only ever refers to the alias name.
#
# Usage: scripts/fetch_data.sh [alias] [bucket] [destination]

set -euo pipefail

ALIAS="${1:-dc26}"
BUCKET="${2:-prc-2026-datasets}"
DEST="${3:-data/raw}"

if ! command -v mc >/dev/null 2>&1; then
  echo "ERROR: MinIO client 'mc' not found. Install it, then run 'mc alias set ${ALIAS} ...'." >&2
  exit 1
fi

if ! mc alias list "${ALIAS}" >/dev/null 2>&1; then
  echo "ERROR: mc alias '${ALIAS}' is not configured." >&2
  echo "Run this yourself (keys are not stored in the repo):" >&2
  echo "  mc alias set ${ALIAS} https://s3.opensky-network.org/ <ACCESS_KEY> <SECRET_KEY>" >&2
  exit 1
fi

mkdir -p "${DEST}"

echo "== Buckets visible to this account =="
mc ls "${ALIAS}"

echo
echo "== Contents of ${BUCKET} =="
mc ls --recursive "${ALIAS}/${BUCKET}"

echo
echo "== Mirroring ${BUCKET} -> ${DEST} =="
# --preserve keeps the remote mtime; mirror is resumable and skips unchanged files.
mc mirror --overwrite --remove=false "${ALIAS}/${BUCKET}/" "${DEST}/"

echo
echo "== Local inventory =="
find "${DEST}" -type f -printf '%10s  %p\n' | sort -k2

echo
echo "== SHA256 manifest -> ${DEST}/SHA256SUMS =="
( cd "${DEST}" && find . -type f ! -name 'SHA256SUMS' -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
cat "${DEST}/SHA256SUMS"

echo
echo "Done. Restricted data stays in ${DEST}/ which is git-ignored."
