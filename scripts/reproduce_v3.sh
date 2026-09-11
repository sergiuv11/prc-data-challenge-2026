#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python environment not found at $PYTHON" >&2
    echo "Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 2
fi

for input in data/raw/ranking.parquet data/raw/submitting.parquet; do
    if [[ ! -s "$input" ]]; then
        echo "ERROR: required private input is missing: $input" >&2
        exit 2
    fi
done
if [[ "$(find data/raw -maxdepth 1 -name 'training_*.parquet' -type f | wc -l)" -ne 12 ]]; then
    echo "ERROR: expected exactly 12 monthly training files under data/raw" >&2
    exit 2
fi

REPORT_ROOT="reports/reproduction_v3"
MODEL_ROOT="models/reproduction_v3"
SUBMISSION_ROOT="submissions/reproduction_v3"
DATA_ROOT="data/reproduction_v3"
CLEAN_ROOT="$DATA_ROOT/clean"
FEATURE_ROOT="$DATA_ROOT/features"
GLOBAL_REPORT="$REPORT_ROOT/global/model.md"
mkdir -p "$REPORT_ROOT"

(cd data/raw && sha256sum ./*.parquet) > "$REPORT_ROOT/input_SHA256SUMS"

echo "[1/10] Normalising the organiser Parquet files"
"$PYTHON" scripts/normalize_parquet.py --src data/raw --dst "$CLEAN_ROOT"

echo "[2/10] Auditing schema and timestamp leakage"
"$PYTHON" scripts/audit_schema.py --data-dir "$CLEAN_ROOT" --out "$REPORT_ROOT/audit"
"$PYTHON" scripts/leak_probe.py --data-dir "$CLEAN_ROOT" --out "$REPORT_ROOT/leak"

echo "[3/10] Building the shared 43-feature tables"
"$PYTHON" scripts/build_features.py --data-dir "$CLEAN_ROOT" --out "$FEATURE_ROOT"

echo "[4/10] Reproducing fixed-tree global validation"
"$PYTHON" scripts/train_model.py \
    --features "$FEATURE_ROOT" \
    --data-dir "$CLEAN_ROOT" \
    --out "$REPORT_ROOT/global" \
    --models-dir "$MODEL_ROOT/global_validation" \
    --report-name model.md \
    --splits composition,forward,seasonal \
    --rounds 436

echo "[5/10] Reproducing the three specialist validation arms"
for split in composition forward seasonal; do
    "$PYTHON" scripts/experiment_airport_specialists.py \
        --features "$FEATURE_ROOT" \
        --data-dir "$CLEAN_ROOT" \
        --baseline "$REPORT_ROOT/global/preds_${split}.parquet" \
        --split "$split" \
        --rounds 436 \
        --out "$REPORT_ROOT/specialists_${split}" \
        --models-dir "$MODEL_ROOT/specialists_${split}"
done

echo "[6/10] Enforcing the composition gate"
"$PYTHON" scripts/compare_predictions.py \
    --baseline "$REPORT_ROOT/global/preds_composition.parquet" \
    --candidate "$REPORT_ROOT/specialists_composition/preds_composition.parquet" \
    --features "$FEATURE_ROOT/train_departures.parquet" \
    --data-dir "$CLEAN_ROOT" \
    --candidate-airports EDDF,EGLL,EHAM \
    --gate-policy shipping \
    --label "50/50 global and airport specialists" \
    --out "$REPORT_ROOT/specialists_composition/comparison.md"
grep -q "Verdict: PASS composition gate" \
    "$REPORT_ROOT/specialists_composition/comparison.md" || {
        echo "ERROR: composition gate failed, final construction refused" >&2
        exit 1
    }

echo "[7/10] Enforcing the forward and seasonal gates"
for split in forward seasonal; do
    "$PYTHON" scripts/compare_specialist_confirmation.py \
        --baseline "$REPORT_ROOT/global/preds_${split}.parquet" \
        --candidate "$REPORT_ROOT/specialists_${split}/preds_${split}.parquet" \
        --features "$FEATURE_ROOT/train_departures.parquet" \
        --data-dir "$CLEAN_ROOT" \
        --split "$split" \
        --candidate-airports EDDF,EGLL,EHAM \
        --label "50/50 global and airport specialists" \
        --out "$REPORT_ROOT/specialists_${split}/comparison.md"
    grep -q "Verdict: PASS" "$REPORT_ROOT/specialists_${split}/comparison.md" || {
        echo "ERROR: $split gate failed, final construction refused" >&2
        exit 1
    }
done

echo "[8/10] Training and validating the complete global V2 foundation"
"$PYTHON" scripts/train_model.py \
    --features "$FEATURE_ROOT" \
    --data-dir "$CLEAN_ROOT" \
    --out "$REPORT_ROOT/final_global" \
    --models-dir "$MODEL_ROOT/final_global" \
    --submissions "$SUBMISSION_ROOT" \
    --team-name jubilant-vase \
    --version 2 \
    --rounds 436 \
    --final-only \
    --make-submission \
    --overlay \
    --gate-report "$GLOBAL_REPORT" \
    --gate-preds "$REPORT_ROOT/global/preds_composition.parquet"

echo "[9/10] Training the final specialists and constructing V3"
"$PYTHON" scripts/build_specialist_submission.py \
    --features "$FEATURE_ROOT" \
    --data-dir "$CLEAN_ROOT" \
    --global-model "$MODEL_ROOT/final_global/lgbm_v2.txt" \
    --v2-submission "$SUBMISSION_ROOT/jubilant-vase_v2.parquet" \
    --out "$SUBMISSION_ROOT/jubilant-vase_v3.parquet" \
    --models-dir "$MODEL_ROOT/final_specialists" \
    --audit "$REPORT_ROOT/final_build_audit.json"

echo "[10/10] Running the final structural validator"
"$PYTHON" scripts/validate_submission.py \
    "$SUBMISSION_ROOT/jubilant-vase_v3.parquet" --data-dir "$CLEAN_ROOT"

echo
echo "PASS: the V3 architecture and gates were rebuilt locally. Nothing was uploaded."
echo "Submission: $SUBMISSION_ROOT/jubilant-vase_v3.parquet"
echo "Audit: $REPORT_ROOT/final_build_audit.json"
