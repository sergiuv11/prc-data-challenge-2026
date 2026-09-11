#!/usr/bin/env bash
set -euo pipefail

readonly ALIAS_NAME="prc2026"
readonly S3_ENDPOINT="https://s3.opensky-network.org"

if ! command -v mc >/dev/null 2>&1; then
    echo "MinIO client 'mc' is not installed." >&2
    exit 1
fi

read -r -p "OpenSky S3 access key: " access_key
read -r -s -p "OpenSky S3 secret key: " secret_key
printf '\n'

if [[ -z "$access_key" || -z "$secret_key" ]]; then
    echo "Both values are required." >&2
    unset access_key secret_key
    exit 1
fi

mc alias set "$ALIAS_NAME" "$S3_ENDPOINT" "$access_key" "$secret_key"
unset access_key secret_key

echo
echo "Access configured outside the repository in the MinIO client config."
echo "Testing the read-only dataset bucket:"
mc ls "$ALIAS_NAME/prc-2026-datasets"
