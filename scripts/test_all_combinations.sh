#!/bin/bash
# Test all backend × nbits combinations for scientific comparison
# Date: Nov 24, 2025

MODEL="gpt2"
CONTEXTS="128 256"
SEED=42

echo "========================================="
echo "Testing all quantization combinations"
echo "Model: $MODEL"
echo "Contexts: $CONTEXTS"
echo "========================================="
echo ""

# 1. FP16 Baseline (already have this)
echo "[1/5] FP16 Baseline (skipping - already exists)"
echo ""

# 2. INT4 + quanto (already have this)
echo "[2/5] INT4 + quanto (skipping - already exists)"
echo ""

# 3. INT8 + hqq (already have this)
echo "[3/5] INT8 + hqq (skipping - already exists)"
echo ""

# 4. INT4 + hqq (NEW)
echo "[4/5] Testing INT4 + hqq..."
python3 quantize_kvcache_hf.py \
    --model $MODEL \
    --context-lengths $CONTEXTS \
    --quantize \
    --nbits 4 \
    --backend hqq \
    --seed $SEED

echo ""
echo "Waiting 3 seconds for CUDA cleanup..."
sleep 3

# 5. INT8 + quanto (NEW)
echo "[5/5] Testing INT8 + quanto..."
python3 quantize_kvcache_hf.py \
    --model $MODEL \
    --context-lengths $CONTEXTS \
    --quantize \
    --nbits 8 \
    --backend quanto \
    --seed $SEED

echo ""
echo "========================================="
echo "✅ All combinations tested!"
echo "========================================="
echo ""
echo "Results in: ../results/raw/"
ls -lh ../results/raw/gpt2_int*
