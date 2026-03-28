#!/bin/bash
# Phase 1 Completion Pipeline
# Runs after download_rvl_forms.py completes

set -e

PROJ_ROOT=/Users/sofiaclaudiabonoan/Desktop/silly-stats
cd "$PROJ_ROOT"

source ./venv/bin/activate

MANIFEST="data/raw/rvl_forms_manifest.csv"
TIMEOUT=3600  # 1 hour

# Wait for download to complete
echo "Waiting for RVL-CDIP form download to complete..."
echo "Timeout: ${TIMEOUT}s"

elapsed=0
while [ ! -f "$MANIFEST" ] && [ $elapsed -lt $TIMEOUT ]; do
    sleep 10
    elapsed=$((elapsed + 10))
    pct=$((elapsed * 100 / TIMEOUT))
    printf "\r[%3d%%] Elapsed: %d/%d seconds" $pct $elapsed $TIMEOUT
done

if [ ! -f "$MANIFEST" ]; then
    echo -e "\n❌ Download timeout after $TIMEOUT seconds"
    exit 1
fi

echo -e "\n✓ Download complete!"
echo "  Forms downloaded: $(ls -1 data/raw/form | wc -l)"
echo ""

# Now run the extraction pipeline
echo "=========================================="
echo "Phase 1: Text Extraction & Preprocessing"
echo "=========================================="
echo ""

echo "Step 1: Extract text via OCR..."
python3 scripts/extract_real_data.py

echo ""
echo "Step 2: Clean text..."
python3 scripts/clean_text.py

echo ""
echo "Step 3: Build train/val/test splits..."
python3 scripts/build_dataset.py

echo ""
echo "Step 4: Generate TF-IDF features..."
python3 scripts/make_features.py

echo ""
echo "=========================================="
echo "✓ Phase 1 Complete!"
echo "=========================================="
echo ""
echo "Next: Run Phase 2 models"
echo "  python3 scripts/train_models.py"
