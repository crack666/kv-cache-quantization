#!/usr/bin/env bash
# NVFP4-Durchsatzmessung unter vLLM.
#
# Beantwortet genau eine Frage: Wie viel schneller ist NVFP4 bei 128k als die
# 80,2 tok/s, die der Ollama-Pfad dort liefert? Alles andere ist bereits geklaert
# -- der Checkpoint ist mit 21,81 GiB groesser als das GGUF und vLLMs KV kommt
# nur bis FP8, der Kontext bleibt also unter dem, was Ollama schafft.
#
# Der Server wird einmal mit --max-model-len 131072 gestartet; kuerzere Kontexte
# laufen darin mit. Die KV-Kapazitaet haengt an --gpu-memory-utilization, nicht
# an der Kontextlaenge, ein zweiter Start waere also verschwendete Ladezeit.
#
# NIE parallel zu einer Ollama-Instanz: 32 GB reichen fuer ein residentes 27B.
#
#   ./run_vllm.sh
#   ./run_vllm.sh --dry-run

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="http://127.0.0.1:11436"
OUT="$HERE/../../results/ollama_probe/raw_vllm"
LOG="$HERE/../../results/ollama_probe/vllm_$(date +%Y%m%d_%H%M%S).log"
CONTEXTS=(32768 131072)

[[ "${1:-}" == "--dry-run" ]] && { echo "[dry] vLLM NVFP4 @ ${CONTEXTS[*]}"; exit 0; }

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "================================================================"
say "vLLM NVFP4. Log: $LOG"
say "Kontexte: ${CONTEXTS[*]}"
say "================================================================"

# Beide Ollama-Instanzen muessen weg -- sonst teilen sich zwei 27B die Karte.
for c in ollama-container ollama-bench; do
  if docker ps --format '{{.Names}}' | grep -qx "$c"; then
    say "ABBRUCH: $c laeuft noch. Erst stoppen."
    exit 1
  fi
done

others=$(powershell.exe -NoProfile -Command \
  "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'bench_ollama|bench_vllm' } | Measure-Object).Count" 2>/dev/null | tr -d '\r')
if [[ "${others:-0}" != "0" ]]; then
  say "ABBRUCH: es laeuft bereits ein Messprozess ($others)."
  exit 1
fi

say "Starte vLLM ..."
docker compose -f "$HERE/compose.vllm.yml" down >/dev/null 2>&1
VLLM_MAX_LEN=131072 docker compose -f "$HERE/compose.vllm.yml" up -d >/dev/null 2>&1

# Ein 22-GiB-Checkpoint plus torch.compile braucht Minuten. Grosszuegig warten,
# aber auf Absturz pruefen -- ein OOM im Graph-Capture ist der erwartbare Fehler.
say "Warte auf Bereitschaft (kann einige Minuten dauern) ..."
ready=0
for i in $(seq 1 90); do
  if curl -sf "$HOST/v1/models" >/dev/null 2>&1; then ready=1; break; fi
  if ! docker ps --format '{{.Names}}' | grep -qx vllm-bench; then
    say "ABBRUCH: Container ist gestorben. Letzte Logzeilen:"
    docker logs --tail 40 vllm-bench 2>&1 | sed 's/^/    /'
    exit 1
  fi
  sleep 10
done

if [[ $ready -eq 0 ]]; then
  say "ABBRUCH: vLLM wurde nicht bereit. Letzte Logzeilen:"
  docker logs --tail 40 vllm-bench 2>&1 | sed 's/^/    /'
  docker compose -f "$HERE/compose.vllm.yml" down >/dev/null 2>&1
  exit 1
fi
say "vLLM bereit."

# Die tatsaechliche KV-Kapazitaet steht im Log und ist die Kennzahl, an der
# haengt, welcher Kontext ueberhaupt geht.
docker logs vllm-bench 2>&1 | grep -iE "KV cache size|GPU KV cache|maximum concurrency|Memory profiling" \
  | tail -5 | sed 's/^/    /'

say "Messung ..."
timeout 10800 python "$HERE/bench_vllm.py" \
  --host "$HOST" --contexts "${CONTEXTS[@]}" \
  --measure-runs 5 --warmup-runs 1 --decode-tokens 512 \
  --output-dir "$OUT" \
  || say "FEHLER/Timeout in der Messung."

docker compose -f "$HERE/compose.vllm.yml" down >/dev/null 2>&1
say "vLLM gestoppt."
say "Fertig um $(date)."
