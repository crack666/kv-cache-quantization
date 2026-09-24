#!/usr/bin/env bash
# Kurzvergleich zweier Ollama-Versionen beim Decode-Durchsatz.
#
# Abkuerzung gegenueber der vollen Harness: der Prompt bleibt zwischen den
# Laeufen gleich, der Prefill trifft also den Cache. Fuer den Decode ist das
# egal -- er wird ueber eval_count/eval_duration gemessen und laeuft in beiden
# Faellen ueber den vollen, gefuellten Kontext. Gespart wird nur die Zeit fuer
# wiederholte 83-Sekunden-Prefills.
#
# Laeuft auf der Bench-Instanz (Port 11435), damit die Produktionsinstanz und
# ihre Clients unberuehrt bleiben. Erwartet, dass ollama-container gestoppt ist.
#
#   ./compare_versions.sh 0.33.0 0.33.2

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="http://127.0.0.1:11435"
MODEL="qwen3.8:27b"
CTX=131072
RUNS=3
DECODE_TOKENS=512

VERSIONS=("$@")
[[ ${#VERSIONS[@]} -lt 1 ]] && { echo "Versionen angeben, z.B. 0.33.0 0.33.2" >&2; exit 2; }

if docker ps --format '{{.Names}}' | grep -qx ollama-container; then
  echo "ABBRUCH: ollama-container laeuft. Erst stoppen -- beide zusammen passen nicht ins VRAM." >&2
  exit 1
fi

echo "Modell:   $MODEL"
echo "Kontext:  $CTX (q8_0)"
echo "Protokoll: 1 Warmup mit vollem Prefill, dann $RUNS Laeufe a $DECODE_TOKENS Decode-Token"
echo

for v in "${VERSIONS[@]}"; do
  echo "=== Ollama $v ==="
  docker compose -f "$HERE/compose.bench.yml" down >/dev/null 2>&1
  OLLAMA_IMAGE="ollama/ollama:$v" OLLAMA_KV_CACHE_TYPE=q8_0 \
    docker compose -f "$HERE/compose.bench.yml" up -d >/dev/null 2>&1

  ready=0
  for _ in $(seq 1 40); do
    curl -sf "$HOST/api/tags" >/dev/null 2>&1 && { ready=1; break; }
    sleep 2
  done
  [[ $ready -eq 0 ]] && { echo "  nicht bereit geworden, uebersprungen"; continue; }

  echo "  Version laut Container: $(docker exec ollama-bench ollama --version 2>&1 | tr -d '\r')"
  python "$HERE/quick_decode.py" --host "$HOST" --model "$MODEL" \
    --context "$CTX" --runs "$RUNS" --decode-tokens "$DECODE_TOKENS"
  echo
done

docker compose -f "$HERE/compose.bench.yml" down >/dev/null 2>&1
echo "Bench-Instanz gestoppt. Produktion zurueckholen: docker start ollama-container"
