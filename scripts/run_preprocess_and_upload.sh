#!/bin/bash
set -euo pipefail

echo "===== PREPROCESS & UPLOAD JOB START ====="
date
hostname
pwd

echo "Python version:"
python --version

echo "===== ENVIRONMENT SETUP ====="

export HF_HOME="$PWD/hf_cache"
export HUGGINGFACE_HUB_CACHE="$PWD/hf_cache"
mkdir -p "$HF_HOME"

# Ensure pip-installed scripts (e.g. aws, huggingface-cli) are on PATH
export PATH="$HOME/.local/bin:$PATH"

echo "Installing Python dependencies"
python -m pip install --no-cache-dir numpy Pillow scipy scikit-image awscli huggingface_hub

# Python's zipfile handles most LZMA zips natively.
# Install p7zip as fallback in case Python's lzma support is incomplete.
if ! which 7z &>/dev/null && ! which 7zz &>/dev/null; then
    echo "Attempting to install p7zip as fallback..."
    (yum install -y p7zip 2>/dev/null || dnf install -y p7zip 2>/dev/null || apt-get install -y p7zip-full 2>/dev/null) \
        && echo "p7zip installed." \
        || echo "WARNING: Could not install p7zip. Will rely on Python zipfile."
fi

echo "Verifying tools..."
aws --version

# ── Configuration ─────────────────────────────────────────────────────────────

# Preprocess to local scratch — this is ephemeral, HF upload is the
# durable storage. Do NOT point this at a non-existent shared path.
OUTPUT_DIR="$PWD/preprocessed"

SPLIT="${SPLIT:-}"
K="${K:-256}"
REG_BLOCK_SIZE="${REG_BLOCK_SIZE:-8}"
OVERLAP_THRESHOLD="${OVERLAP_THRESHOLD:-0.45}"
USE_DENSE_FLOW="${USE_DENSE_FLOW:-true}"
FLOW_ATTACHMENT="${FLOW_ATTACHMENT:-15}"
FLOW_TIGHTNESS="${FLOW_TIGHTNESS:-0.3}"
NUM_WARP="${NUM_WARP:-5}"
NUM_WORKERS="${NUM_WORKERS:-0}"

# Upload is required — HF is how the data survives after the job ends
REPO_ID="${REPO_ID:?ERROR: REPO_ID not set. Example: your-username/spc-preprocessed}"
HF_TOKEN="${HF_TOKEN:?ERROR: HF_TOKEN not set. Get one at https://huggingface.co/settings/tokens}"
PRIVATE="${PRIVATE:-false}"

export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

echo "OUTPUT_DIR:         $OUTPUT_DIR (local scratch, ephemeral)"
echo "SPLIT:              ${SPLIT:-all}"
echo "K:                  $K"
echo "REG_BLOCK_SIZE:     $REG_BLOCK_SIZE"
echo "OVERLAP_THRESHOLD:  $OVERLAP_THRESHOLD"
echo "USE_DENSE_FLOW:     $USE_DENSE_FLOW"
echo "FLOW_ATTACHMENT:    $FLOW_ATTACHMENT"
echo "FLOW_TIGHTNESS:     $FLOW_TIGHTNESS"
echo "NUM_WARP:           $NUM_WARP"
echo "NUM_WORKERS:        $NUM_WORKERS"
echo "REPO_ID:            $REPO_ID"

# ── Preprocess ────────────────────────────────────────────────────────────────

SPLIT_ARG=""
if [ -n "$SPLIT" ]; then
    SPLIT_ARG="--split $SPLIT"
fi

DENSE_FLOW_ARG=""
if [ "$USE_DENSE_FLOW" = "false" ]; then
    DENSE_FLOW_ARG="--no-dense-flow"
fi

echo "===== START PREPROCESSING ====="
python preprocess_full_dataset.py \
    --output-dir "$OUTPUT_DIR" \
    --scratch-dir "$PWD/_scratch" \
    --K "$K" \
    --reg-block-size "$REG_BLOCK_SIZE" \
    --overlap-threshold "$OVERLAP_THRESHOLD" \
    --flow-attachment "$FLOW_ATTACHMENT" \
    --flow-tightness "$FLOW_TIGHTNESS" \
    --num-warp "$NUM_WARP" \
    --num-workers "$NUM_WORKERS" \
    $DENSE_FLOW_ARG \
    $SPLIT_ARG

echo "===== PREPROCESSING COMPLETE ====="
date

# ── Upload to HuggingFace ────────────────────────────────────────────────────

PRIVATE_ARG=""
if [ "$PRIVATE" = "true" ]; then
    PRIVATE_ARG="--private"
fi

echo "===== START UPLOAD ====="
python upload_to_hf.py \
    --dataset-dir "$OUTPUT_DIR" \
    --repo-id "$REPO_ID" \
    $PRIVATE_ARG

echo "===== UPLOAD COMPLETE ====="
echo "Dataset available at: https://huggingface.co/datasets/$REPO_ID"
echo "===== JOB FINISHED ====="
date