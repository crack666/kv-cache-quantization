#!/bin/bash
# Complete Mistral-7B profiling with all configs
# Runs: FP16, INT8, INT4, INT2 across contexts 128-4096

set -e

MODEL="mistralai/Mistral-7B-v0.1"
CONTEXTS="128 256 512 1024 2048 4096"

echo "========================================="
echo "Mistral-7B Complete Profiling"
echo "========================================="
echo "Model: $MODEL"
echo "Contexts: $CONTEXTS"
echo ""

# FP16 Baseline
echo "[1/4] Running FP16 baseline..."
python scripts/quantize_kvcache_hf.py \
    --model "$MODEL" \
    --context-lengths $CONTEXTS \
    --no-quantize \
    --seed 42 \
    --output-dir results/raw

# INT8 HQQ
echo "[2/4] Running INT8 HQQ..."
python scripts/quantize_kvcache_hf.py \
    --model "$MODEL" \
    --context-lengths $CONTEXTS \
    --quantize \
    --nbits 8 \
    --backend hqq \
    --seed 42 \
    --output-dir results/raw

# INT4 HQQ
echo "[3/4] Running INT4 HQQ..."
python scripts/quantize_kvcache_hf.py \
    --model "$MODEL" \
    --context-lengths $CONTEXTS \
    --quantize \
    --nbits 4 \
    --backend hqq \
    --seed 42 \
    --output-dir results/raw

# INT2 HQQ
echo "[4/4] Running INT2 HQQ..."
python scripts/quantize_kvcache_hf.py \
    --model "$MODEL" \
    --context-lengths $CONTEXTS \
    --quantize \
    --nbits 2 \
    --backend hqq \
    --seed 42 \
    --output-dir results/raw

echo ""
echo "✅ All Mistral-7B measurements complete!"
echo "Results saved to: results/raw/profile_mistral_*.json"
