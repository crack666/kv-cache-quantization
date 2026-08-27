#!/usr/bin/env bash
# Kontext-Kurve: wo liegt der beste Arbeitspunkt zwischen 128k und 256k?
#
# 262144 ist eine willkuerliche Marke (das native Fenster), nicht zwingend der
# beste Betriebspunkt. Aus den bisherigen zwei Messpunkten folgt linear:
#
#     Gesamtbelegung ≈ 17797 MiB + 29,1 KiB x Kontextlaenge
#
# (29,1 statt der reinen 18 KiB des q4_0-KV, weil die Compute-Puffer mitwachsen.)
# Damit passt bei belebtem Desktop und Embedder nur noch rund 232k, mit 1 GB
# Reserve etwa 196k, mit 2 GB etwa 160k. Der interessante Bereich liegt also
# zwischen 128k und 200k -- und genau dort fehlen Messpunkte.
#
# Protokoll gegenueber den bisherigen Laeufen geaendert: --decode-tokens 512
# statt 128. Die alten Zellen massen nur 1-2 Sekunden Decode, was die Mediane
# dem Zufall der MTP-Akzeptanzrate auslieferte (Streuung bis 29 %). Deshalb
# sind die Ergebnisse dieses Laufs NICHT mit den 128-Token-Zellen mischbar;
# er misst die Bezugsgroesse 131072 selbst noch einmal mit.
#
#   ./run_curve.sh              # alles
#   ./run_curve.sh --dry-run

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="http://127.0.0.1:11435"
OUT="$HERE/../../results/ollama_probe/raw"
LOG="$HERE/../../results/ollama_probe/curve_$(date +%Y%m%d_%H%M%S).log"
IMAGE="${OLLAMA_IMAGE:-ollama/ollama:0.33.0}"
CONTAINER="ollama-bench"
MODEL="qwen3.8:27b"          # Produktivstand; UD erst, wenn ein Wechsel ansteht
DECODE_TOKENS=512

DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "================================================================"
say "Kontext-Kurve. Log: $LOG"
say "Modell: $MODEL | decode-tokens: $DECODE_TOKENS"
say "================================================================"

if docker ps --format '{{.Names}}' | grep -qx 'ollama-container'; then
  say "ABBRUCH: Produktions-Ollama laeuft."
  exit 1
fi

# Lehre aus dem Doppellauf: ein zweiter Messprozess konfiguriert den Container
# unter diesem hier um, und die KV-Beschriftung wird still falsch.
others=$(powershell.exe -NoProfile -Command \
  "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'bench_ollama' } | Measure-Object).Count" 2>/dev/null | tr -d '\r')
if [[ "${others:-0}" != "0" ]]; then
  say "ABBRUCH: es laeuft bereits ein bench_ollama.py ($others Prozesse)."
  exit 1
fi

if [[ $DRY -eq 1 ]]; then
  say "[dry] q4_0 @ 131072 163840 196608 262144"
  say "[dry] q8_0 @ 131072"
  say "[dry] Budgetmessung (Embedder + Desktop)"
  exit 0
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

run_cell() {  # $1=kv $2=tag $3...=kontexte
  local kv="$1" tag="$2"; shift 2
  say "  >>> $MODEL @ kv=$kv, ctx=$*"
  timeout 21600 python "$HERE/bench_ollama.py" \
    --model "$MODEL" --kv-cache-type "$kv" --host "$HOST" \
    --contexts "$@" --tag "$tag" --container "$CONTAINER" \
    --measure-runs 5 --warmup-runs 1 --decode-tokens "$DECODE_TOKENS" \
    --output-dir "$OUT" \
    || say "  FEHLER/Timeout bei kv=$kv -- weiter."
}

# --- Budgetposten zuerst: schnell und unabhaengig vom Rest ----------------
say "Schritt 1/3: Budgetposten messen (Embedder, Desktop)"
start_bench q4_0 || exit 1
python "$HERE/measure_budget.py" --host "$HOST" \
  --output "$OUT/../budget_$(date +%Y%m%d_%H%M%S).json" \
  || say "  Budgetmessung fehlgeschlagen -- weiter."

# --- Die Kurve ------------------------------------------------------------
say "Schritt 2/3: q4_0 ueber die Kontext-Kurve"
run_cell q4_0 "curve" 131072 163840 196608 262144

# --- q8_0 nur als Vergleichspunkt bei 128k --------------------------------
# Bei laengerem Kontext scheidet q8_0 ohnehin aus: 8704 MiB KV bei 256k plus
# Embedder sprengen das Budget. Der Vergleich ist nur bei 128k interessant.
say "Schritt 3/3: q8_0 als Vergleichspunkt bei 131072"
start_bench q8_0 || exit 1
run_cell q8_0 "curve" 131072

docker compose -f "$HERE/compose.bench.yml" down >/dev/null 2>&1
say "Bench-Instanz gestoppt."
python "$HERE/validate_runs.py" || say "WARNUNG: Validierung meldet Fehlzuordnungen!"
say "Fertig um $(date)."
