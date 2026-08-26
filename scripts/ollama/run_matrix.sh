#!/usr/bin/env bash
# Fahrt die volle Matrix Gewichts-Quant x KV-Cache-Typ x Kontextlaenge.
#
# Der KV-Cache-Typ ist eine Container-Env-Variable, also wird die Bench-Instanz
# je KV-Stufe einmal neu gestartet. Die Gewichts-Quantisierung steckt dagegen im
# Modell-Tag und wechselt ohne Neustart.
#
# Voraussetzung: die Produktions-Ollama-Instanz (Port 11434) ist gestoppt.
# Beide gleichzeitig wuerden sich die 32 GB VRAM teilen.
#
#   ./run_matrix.sh                       # Track A: qwen3.8-27b, volle Matrix
#   ./run_matrix.sh --track-b             # Bruecke zur Arbeit (f16-Gewichte)
#   ./run_matrix.sh --dry-run             # nur zeigen, was liefe

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="http://127.0.0.1:11435"
OUT="$HERE/../../results/ollama_probe/raw"
DRY=0
TRACK="A"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --track-b) TRACK="B" ;;
    --track-a) TRACK="A" ;;
    *) echo "Unbekannte Option: $1" >&2; exit 2 ;;
  esac
  shift
done

if [[ "$TRACK" == "A" ]]; then
  # Track A: das produktive Modell.
  #
  # Erster Eintrag ist der Produktivstand aus der Ollama-Library (Standard-
  # q4_K_M, 16.52 GiB). Die drei folgenden bilden die eigentliche Vergleichs-
  # achse und stammen geschlossen von unsloth (UD-Quants, imatrix-kalibriert),
  # damit ueber die Bitbreiten hinweg dasselbe Quantisierungsverfahren wirkt.
  # Library- und UD-Quants duerfen nicht als eine Achse gelesen werden.
  #
  # q8_0 fehlt bewusst: ~27 GiB Gewichte plus KV sprengen bei langem Kontext
  # die 32 GB. Ueber import_gguf_quants.sh --with-q8 nachruestbar, dann aber
  # nur fuer die kurzen Kontexte auswertbar.
  MODELS=(
    "qwen3.8:27b"
    "qwen3.8-ud:27b-q4_K_M"
    "qwen3.8-ud:27b-q5_K_M"
    "qwen3.8-ud:27b-q6_K"
  )
  CONTEXTS=(8192 32768 65536 131072)
  TAG="trackA"
else
  # Track B: Bruecke zu den Zahlen der Arbeit. Gewichte in fp16 wie dort, damit
  # sich allein die KV-Implementierung unterscheidet (llama.cpp statt quanto/HQQ).
  #
  # qwen3:8b-fp16 ist exakt ein Modell des Thesis-Testfelds (Qwen3-8B,
  # Key-Kurtosis 23.01, dort fragil: bricht bei INT2 ein). Damit laesst sich
  # pruefen, ob der llama.cpp-Quantisierer denselben Schaden anrichtet wie
  # quanto/HQQ -- die Voraussetzung dafuer, Track A ueberhaupt gegen die
  # Ergebnisse der Arbeit zu halten.
  #
  # Mistral steht in der Library nur als 7b-text-fp16 (Basismodell, aber nicht
  # v0.1 wie in der Arbeit) zur Verfuegung. Als zweiter Stuetzpunkt brauchbar,
  # der Versionsunterschied gehoert aber in die Auswertung.
  MODELS=("qwen3:8b-fp16" "mistral:7b-text-fp16")
  CONTEXTS=(4096 8192 16384 32768)
  TAG="trackB"
fi

KV_TYPES=("f16" "q8_0" "q4_0")

echo "==================================================================="
echo "Track $TRACK"
echo "Modelle:   ${MODELS[*]}"
echo "KV-Typen:  ${KV_TYPES[*]}"
echo "Kontexte:  ${CONTEXTS[*]}"
echo "Ausgabe:   $OUT"
echo "==================================================================="

# Produktionsinstanz darf nicht laufen.
if docker ps --format '{{.Names}}' | grep -qx 'ollama-container'; then
  echo "ABBRUCH: ollama-container (Produktion, Port 11434) laeuft noch." >&2
  echo "         Zuerst stoppen:  docker stop ollama-container" >&2
  exit 1
fi

wait_ready() {
  for _ in $(seq 1 60); do
    if curl -sf "$HOST/api/tags" >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  echo "ABBRUCH: Bench-Instanz wurde nicht bereit." >&2
  return 1
}

for kv in "${KV_TYPES[@]}"; do
  echo
  echo "--- KV-Cache-Typ: $kv -----------------------------------------"
  if [[ $DRY -eq 1 ]]; then
    for m in "${MODELS[@]}"; do
      echo "  [dry] $m  kv=$kv  ctx=${CONTEXTS[*]}"
    done
    continue
  fi

  docker compose -f "$HERE/compose.bench.yml" down >/dev/null 2>&1 || true
  OLLAMA_KV_CACHE_TYPE="$kv" docker compose -f "$HERE/compose.bench.yml" up -d
  wait_ready

  for m in "${MODELS[@]}"; do
    echo
    echo ">>> $m @ kv=$kv"
    # Fehlt das Tag lokal, einmalig ziehen. UD-Quants stehen nicht in der
    # Ollama-Library, die kommen ueber import_gguf_quants.sh von HuggingFace.
    if ! curl -sf "$HOST/api/tags" | grep -q "\"$m\""; then
      if [[ "$m" == qwen3.8-ud:* ]]; then
        echo "    ABBRUCH: $m nicht registriert. Erst importieren:" >&2
        echo "             ./import_gguf_quants.sh" >&2
        exit 1
      fi
      echo "    Tag nicht lokal -- ziehe $m ..."
      docker exec ollama-bench ollama pull "$m"
    fi
    python "$HERE/bench_ollama.py" \
      --model "$m" \
      --kv-cache-type "$kv" \
      --host "$HOST" \
      --contexts "${CONTEXTS[@]}" \
      --tag "$TAG" \
      --output-dir "$OUT" || echo "    Lauf fehlgeschlagen, weiter mit naechster Zelle." >&2
  done
done

if [[ $DRY -eq 0 ]]; then
  docker compose -f "$HERE/compose.bench.yml" down >/dev/null 2>&1 || true
  echo
  echo "Bench-Instanz gestoppt. Produktionsinstanz wieder starten mit:"
  echo "  docker start ollama-container"
fi
