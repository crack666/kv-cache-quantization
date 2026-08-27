#!/usr/bin/env bash
# Gekuerzter Lauf mit korrigierten Modellen.
#
# Vorgeschichte: Der erste Nachtlauf hat die UD-Modelle unfair gemessen. Ihr
# Modelfile enthielt nur FROM <gguf> -- ohne Vision-Projektor und ohne
# PARAMETER draft_num_predict, mit dem die Library-Variante spekulatives
# Decoding ueber den MTP-Kopf faehrt (~3,6 akzeptierte Token je Draft). Der
# MTP-Kopf steckt in beiden GGUFs, beide haben 866 Tensoren einschliesslich
# blk.64.nextn.* -- er war beim UD-Import nur nie eingeschaltet.
#
# Beide Modelle laufen hier deshalb mit identischer Ausstattung: Vision,
# Renderer, Parser, draft_num_predict 4.
#
# Zwei Abschnitte:
#
#   A  Decode-Nachmessung bei 32k/128k mit 5 statt 2 Messlaeufen. Der erste
#      Lauf stuetzte den Decode-Median auf nur zwei Werte; an diesem Befund
#      haengt die Modellwahl, er soll belastbar sein.
#   B  Nativer 256k-Kontext. Die eigentliche Frage: haelt die Qualitaet, und
#      was kostet der laengere Kontext an Durchsatz?
#
# f16 faellt weg -- 8 GiB KV bei 128k, 16 GiB bei 256k passen mit keinem
# Gewicht. In allen 21 Zellen des ersten Laufs war q8_0 gegenueber f16
# verlustfrei.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="http://127.0.0.1:11435"
OUT="$HERE/../../results/ollama_probe/raw"
LOG="$HERE/../../results/ollama_probe/targeted_$(date +%Y%m%d_%H%M%S).log"
IMAGE="${OLLAMA_IMAGE:-ollama/ollama:0.33.0}"
CONTAINER="ollama-bench"

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
say() { echo "[$(date +%H:%M:%S)] $*"; }

# Library-Variante und UD-Variante, beide voll ausgestattet.
MODELS=("qwen3.8:27b" "qwen3.8-udv:27b-q4_K_M")

say "================================================================"
say "Gekuerzter Lauf mit korrigierten Modellen. Log: $LOG"
say "Modelle: ${MODELS[*]}"
say "================================================================"

if docker ps --format '{{.Names}}' | grep -qx 'ollama-container'; then
  say "ABBRUCH: Produktions-Ollama laeuft."
  exit 1
fi

start_bench() {
  docker compose -f "$HERE/compose.bench.yml" down >/dev/null 2>&1
  OLLAMA_IMAGE="$IMAGE" OLLAMA_KV_CACHE_TYPE="$1" \
    docker compose -f "$HERE/compose.bench.yml" up -d >/dev/null 2>&1
  for _ in $(seq 1 60); do
    curl -sf "$HOST/api/tags" >/dev/null 2>&1 && return 0
    sleep 2
  done
  say "  WARNUNG: Bench-Instanz nicht bereit (KV=$1)."
  return 1
}

run_cell() {  # $1=modell $2=kv $3=tag $4=measure-runs $5...=kontexte
  local model="$1" kv="$2" tag="$3" runs="$4"; shift 4
  if ! curl -sf "$HOST/api/tags" | grep -q "\"$model\""; then
    say "  UEBERSPRUNGEN: $model nicht registriert."
    return
  fi
  say "  >>> $model @ kv=$kv, ctx=$*, ${runs} Messlaeufe"
  timeout 10800 python "$HERE/bench_ollama.py" \
    --model "$model" --kv-cache-type "$kv" --host "$HOST" \
    --contexts "$@" --tag "$tag" --container "$CONTAINER" \
    --measure-runs "$runs" --warmup-runs 1 --output-dir "$OUT" \
    || say "  FEHLER/Timeout bei $model kv=$kv -- weiter."
}

# --- A: Decode-Nachmessung ------------------------------------------------
say "Abschnitt A: Decode-Nachmessung bei 32k/128k, 5 Messlaeufe"
start_bench q8_0 || exit 1
for m in "${MODELS[@]}"; do
  run_cell "$m" q8_0 "fix" 5 32768 131072
done
say "Abschnitt A abgeschlossen."

# --- B: nativer 256k-Kontext ---------------------------------------------
say "Abschnitt B: nativer 256k-Kontext"
for kv in q4_0 q8_0; do
  say "--- KV=$kv @ 256k ---"
  start_bench "$kv" || continue
  for m in "${MODELS[@]}"; do
    run_cell "$m" "$kv" "fix256" 3 262144
  done
done
say "Abschnitt B abgeschlossen."

docker compose -f "$HERE/compose.bench.yml" down >/dev/null 2>&1
say "Bench-Instanz gestoppt. Ergebnisdateien: $(ls -1 "$OUT"/*.json 2>/dev/null | wc -l)"
say "Produktions-Ollama zurueckholen mit: docker start ollama-container"
say "Fertig um $(date)."
