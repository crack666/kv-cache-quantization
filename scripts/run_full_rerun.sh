#!/usr/bin/env bash
# Voller Re-Run der Long-Context-Suite mit korrigiertem Messprotokoll (2026-07-17).
# Protokoll-Fixes: logits_to_keep=1 (Prefill), keine Decode-Cache-Kopien,
# Watchdog gegen PCIe-Swap-Hänger. Gemma jetzt ebenfalls bis 32k (Fairness).
# Alte Ergebnisse bleiben in results/raw/long_context/ (dokumentieren das
# Legacy-Protokoll); neue Ergebnisse nach results/raw/long_context_v2/.
set -uo pipefail

SCRIPTS_DIR="/mnt/a/Users/crack.crackdesk/Documents/Beuth it up/Master 3/Wissenschaftliches Projekt/kv-cache-quantization/scripts"
cd "$SCRIPTS_DIR"
source /home/crack/.venvs/global/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8

OUT=../results/raw/long_context_v2
mkdir -p "$OUT"

MODELS=(
  "mistralai/Mistral-7B-v0.1|mistral_7b"
  "01-ai/Yi-1.5-9B|yi_9b"
  "Qwen/Qwen3-8B|qwen3_8b"
  "Qwen/Qwen2-7B|qwen2_7b"
  "google/gemma-4-E4B|gemma_e4b"
)

total_start=$(date +%s)
for entry in "${MODELS[@]}"; do
  model="${entry%%|*}"
  tag="${entry##*|}"
  model_start=$(date +%s)
  echo ""
  echo "############################################################"
  echo "### $model  (Start: $(date '+%H:%M:%S'))"
  echo "############################################################"
  python profiler_suite.py \
    --model "$model" \
    --attn-backend sdpa \
    --kv-quant none int8-hqq int4-hqq int2-hqq int2-hqq-kivi \
    --context-lengths 4096 8192 16384 32768 \
    --benchmarks ppl needle \
    --warmup-runs 2 --measure-runs 5 --decode-measure-runs 1 \
    --decode-tokens 128 --seed 42 \
    --output-dir "$OUT/" \
    --summary-file "$OUT/${tag}_summary.json"
  model_end=$(date +%s)
  dur=$(( model_end - model_start ))
  echo "### $model fertig ($(date '+%H:%M:%S')) — Laufzeit $(( dur / 60 ))m$(( dur % 60 ))s"
done

total_end=$(date +%s)
echo ""
echo "============================================================"
echo "RE-RUN KOMPLETT: $(( (total_end - total_start) / 60 )) min"
echo "Summaries: $OUT/*_summary.json"
echo "============================================================"
