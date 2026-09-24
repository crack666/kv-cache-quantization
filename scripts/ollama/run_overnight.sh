#!/usr/bin/env bash
# Unbeaufsichtigter Gesamtlauf: Download, Registrierung, Messmatrix.
#
# Zwei Phasen, bewusst in dieser Reihenfolge:
#
#   Phase 1  4 Gewichte x 3 KV-Typen x {8k, 32k, 128k}   -- rund 4,5 h
#   Phase 2  4 Gewichte x q4_0/q8_0 x 256k               -- rund 2,5 h
#
# Phase 1 beantwortet die eigentliche Frage nach dem Arbeitspunkt und laeuft
# deshalb zuerst. Bricht die Nacht vorher ab, sind ihre Ergebnisse trotzdem
# vollstaendig. Phase 2 klaert nur noch, ob sich der native 256k-Kontext lohnt.
#
# Einzelne Zellen duerfen scheitern, ohne den Lauf zu beenden: OOM ist hier ein
# Messergebnis und markiert die Budgetgrenze.
#
#   ./run_overnight.sh              # alles
#   ./run_overnight.sh --phase1     # nur Phase 1
#   ./run_overnight.sh --dry-run

set -uo pipefail   # kein -e: der Lauf soll Einzelfehler ueberleben

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="http://127.0.0.1:11435"
OUT="$HERE/../../results/ollama_probe/raw"
LOG="$HERE/../../results/ollama_probe/overnight_$(date +%Y%m%d_%H%M%S).log"
IMAGE="${OLLAMA_IMAGE:-ollama/ollama:0.33.0}"
CONTAINER="ollama-bench"

PHASE1=1; PHASE2=1; DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase1) PHASE2=0 ;;
    --phase2) PHASE1=0 ;;
    --dry-run) DRY=1 ;;
    *) echo "Unbekannte Option: $1" >&2; exit 2 ;;
  esac
  shift
done

mkdir -p "$OUT" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

say() { echo "[$(date +%H:%M:%S)] $*"; }

MODELS=(
  "qwen3.8:27b"                # Library-q4_K_M: Referenz auf den Produktivstand
  "qwen3.8-ud:27b-q4_K_M"      # ab hier die unsloth-Leiter, ein Rezept
  "qwen3.8-ud:27b-q5_K_M"
  "qwen3.8-ud:27b-q6_K"
)

say "================================================================"
say "Nachtlauf. Log: $LOG"
say "Image: $IMAGE"
say "Modelle: ${MODELS[*]}"
say "================================================================"

# --- Vorbedingungen ------------------------------------------------------
if docker ps --format '{{.Names}}' | grep -qx 'ollama-container'; then
  say "ABBRUCH: Produktions-Ollama laeuft. Beide Instanzen wuerden sich die 32 GB teilen."
  exit 1
fi

if [[ $DRY -eq 1 ]]; then
  say "[dry] Phase 1: ${#MODELS[@]} Modelle x 3 KV x 3 Kontexte = $(( ${#MODELS[@]} * 9 )) Zellen"
  say "[dry] Phase 2: ${#MODELS[@]} Modelle x 2 KV x 1 Kontext  = $(( ${#MODELS[@]} * 2 )) Zellen"
  exit 0
fi

# --- Download (idempotent, ueberspringt vollstaendige Dateien) ------------
say "Schritt 1/3: GGUF-Leiter sicherstellen ..."
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -- bash -lc \
  "bash '/mnt/d/Rettung_A/01_Dokumente/Beuth it up/Master 3/Wissenschaftliches Projekt/kv-cache-quantization/scripts/ollama/download_quants.sh'" \
  2>&1 | tr -d '\0'
say "Download abgeschlossen (Exit $?)."

# --- Container + Registrierung -------------------------------------------
start_bench() {  # $1 = KV-Typ
  docker compose -f "$HERE/compose.bench.yml" down >/dev/null 2>&1
  OLLAMA_IMAGE="$IMAGE" OLLAMA_KV_CACHE_TYPE="$1" \
    docker compose -f "$HERE/compose.bench.yml" up -d >/dev/null 2>&1
  for _ in $(seq 1 60); do
    curl -sf "$HOST/api/tags" >/dev/null 2>&1 && return 0
    sleep 2
  done
  say "  WARNUNG: Bench-Instanz wurde nicht bereit (KV=$1)."
  return 1
}

say "Schritt 2/3: GGUF-Dateien als Ollama-Modelle registrieren ..."
start_bench q8_0 || exit 1
for entry in \
  "Qwen3.8-27B-UD-Q4_K_M.gguf|qwen3.8-ud:27b-q4_K_M" \
  "Qwen3.8-27B-UD-Q5_K_M.gguf|qwen3.8-ud:27b-q5_K_M" \
  "Qwen3.8-27B-UD-Q6_K.gguf|qwen3.8-ud:27b-q6_K"
do
  file="${entry%%|*}"; tag="${entry##*|}"
  if curl -sf "$HOST/api/tags" | grep -q "\"$tag\""; then
    say "  $tag bereits registriert."
    continue
  fi
  say "  registriere $tag ..."
  docker exec "$CONTAINER" sh -c \
    "printf 'FROM /root/.ollama/gguf-import/%s\nPARAMETER temperature 0.2\n' '$file' > /tmp/Mf && ollama create '$tag' -f /tmp/Mf && rm -f /tmp/Mf" \
    || say "  FEHLER bei $tag -- wird uebersprungen."
done

# --- Messung --------------------------------------------------------------
run_cell() {  # $1=modell $2=kv $3=tag $4...=kontexte
  local model="$1" kv="$2" tag="$3"; shift 3
  if ! curl -sf "$HOST/api/tags" | grep -q "\"$model\""; then
    say "  UEBERSPRUNGEN: $model nicht registriert."
    return
  fi
  say "  >>> $model @ kv=$kv, ctx=$*"
  timeout 7200 python "$HERE/bench_ollama.py" \
    --model "$model" --kv-cache-type "$kv" --host "$HOST" \
    --contexts "$@" --tag "$tag" --container "$CONTAINER" \
    --measure-runs 2 --warmup-runs 1 --output-dir "$OUT" \
    || say "  FEHLER/Timeout bei $model kv=$kv -- weiter."
}

if [[ $PHASE1 -eq 1 ]]; then
  say "Schritt 3/3, Phase 1: Matrix ueber 8k/32k/128k"
  for kv in f16 q8_0 q4_0; do
    say "--- KV=$kv ---"
    start_bench "$kv" || continue
    for m in "${MODELS[@]}"; do
      run_cell "$m" "$kv" "p1" 8192 32768 131072
    done
  done
  say "Phase 1 abgeschlossen."
fi

if [[ $PHASE2 -eq 1 ]]; then
  # f16 faellt weg: 8 GiB KV bei 256k passen mit keinem der Gewichte ins Budget.
  say "Phase 2: nativer 256k-Kontext"
  for kv in q4_0 q8_0; do
    say "--- KV=$kv @ 256k ---"
    start_bench "$kv" || continue
    for m in "${MODELS[@]}"; do
      run_cell "$m" "$kv" "p2" 262144
    done
  done
  say "Phase 2 abgeschlossen."
fi

# --- Abschluss ------------------------------------------------------------
docker compose -f "$HERE/compose.bench.yml" down >/dev/null 2>&1
say "Bench-Instanz gestoppt."
say "Ergebnisdateien: $(ls -1 "$OUT"/*.json 2>/dev/null | wc -l)"
say "GPU: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
say "Produktions-Ollama bleibt gestoppt (war es beim Start auch)."
say "Zurueckholen mit: docker start ollama-container"
say "Fertig um $(date)."
