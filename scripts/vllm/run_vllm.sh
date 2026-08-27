#!/usr/bin/env bash
# Gegenmessung unter vLLM: zwei Quantisierungsverfahren, ein Runtime.
#
# Verglichen werden zwei Checkpoints, die beide den MTP-Draft-Kopf mitbringen --
# ohne den kein spekulatives Decoding, und daher kaeme der erhoffte Vorsprung:
#
#   unsloth NVFP4    21,81 GiB   mixed precision (U8/F8_E4M3/BF16, ~6,4 Bit)
#   RedHatAI INT4    18,12 GiB   compressed-tensors, offenbar textonly
#
# Zum Vergleich das GGUF unter Ollama: 15,33 GiB. Beide vLLM-Checkpoints sind
# also groesser, und das frisst genau den Headroom, den Jarvis und fish-speech
# spaeter brauchen. Die offene Frage ist allein der Durchsatz: 79,5 tok/s
# liefert Ollama bei 131072 -- lohnt der Stack-Wechsel dagegen?
#
# NIE parallel zu einer Ollama-Instanz: 32 GB reichen fuer ein residentes 27B.
#
#   ./run_vllm.sh                 # beide Checkpoints
#   ./run_vllm.sh --only redhat   # nur einer
#   ./run_vllm.sh --dry-run

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="http://127.0.0.1:11436"
OUT="$HERE/../../results/ollama_probe/raw_vllm"
LOG="$HERE/../../results/ollama_probe/vllm_$(date +%Y%m%d_%H%M%S).log"
CONTEXTS=(32768 131072)

# Verzeichnisname unter /models | Kennung | angezeigter Modellname
CHECKPOINTS=(
  "Qwen3.8-27B-NVFP4|nvfp4|qwen3.8-nvfp4"
  "Qwen3.8-27B-INT4|int4|qwen3.8-int4"
)

ONLY=""
DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) ONLY="${2:?}"; shift ;;
    --dry-run) DRY=1 ;;
    *) echo "Unbekannte Option: $1" >&2; exit 2 ;;
  esac
  shift
done

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "================================================================"
say "vLLM-Gegenmessung. Log: $LOG"
say "Kontexte: ${CONTEXTS[*]}"
say "================================================================"

if [[ $DRY -eq 1 ]]; then
  for e in "${CHECKPOINTS[@]}"; do
    IFS='|' read -r dir label served <<< "$e"
    [[ -n "$ONLY" && "$ONLY" != "$label" ]] && continue
    say "[dry] $dir ($label) @ ${CONTEXTS[*]}"
  done
  exit 0
fi

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

for entry in "${CHECKPOINTS[@]}"; do
  IFS='|' read -r dir label served <<< "$entry"
  [[ -n "$ONLY" && "$ONLY" != "$label" ]] && continue

  say ""
  say "=== $dir ($label) ==="

  # CUDA-Graphs zuerst versuchen. Die vLLM-Recipe verlangt --enforce-eager auf
  # einer einzelnen 5090, aber das gilt fuer NVFP4 mit 21,81 GiB. Bei INT4
  # (17,95 GiB) passen die Graphs -- und ohne sie kostet jede Decode-Iteration
  # den vollen Kernel-Launch-Overhead: gemessen 45 statt 133 tok/s.
  ready=0
  mode=""
  for attempt in graphs eager; do
    if [[ "$attempt" == "graphs" ]]; then
      files=(-f "$HERE/compose.vllm.yml" -f "$HERE/compose.vllm.graphs.yml")
    else
      files=(-f "$HERE/compose.vllm.yml")
    fi

    docker compose -f "$HERE/compose.vllm.yml" down >/dev/null 2>&1
    VLLM_CKPT="$dir" VLLM_SERVED="$served" VLLM_MAX_LEN=131072       docker compose "${files[@]}" up -d >/dev/null 2>&1

    say "  Start mit $attempt ..."
    for i in $(seq 1 120); do
      if curl -sf "$HOST/v1/models" >/dev/null 2>&1; then ready=1; mode="$attempt"; break; fi
      if ! docker ps --format '{{.Names}}' | grep -qx vllm-bench; then
        say "  Container gestorben ($attempt):"
        docker logs --tail 15 vllm-bench 2>&1 | grep -iE "error|oom|memory" | tail -5 | sed 's/^/    /'
        break
      fi
      sleep 10
    done
    [[ $ready -eq 1 ]] && break
    say "  $attempt gescheitert, naechster Versuch."
  done

  if [[ $ready -eq 0 ]]; then
    say "  UEBERSPRUNGEN: $label wurde in keiner Variante bereit."
    continue
  fi
  say "  bereit ($mode)."

  # Die tatsaechliche KV-Kapazitaet steht im Log und entscheidet, welcher
  # Kontext ueberhaupt moeglich ist.
  docker logs vllm-bench 2>&1 \
    | grep -iE "KV cache size|GPU KV cache|maximum concurrency|Memory profiling|graph capturing" \
    | tail -6 | sed 's/^/    /'

  timeout 10800 python "$HERE/bench_vllm.py" \
    --host "$HOST" --model "$served" --label "${label}-${mode}" \
    --contexts "${CONTEXTS[@]}" \
    --measure-runs 5 --warmup-runs 1 --decode-tokens 512 \
    --output-dir "$OUT" \
    || say "  FEHLER/Timeout bei $label."
done

docker compose -f "$HERE/compose.vllm.yml" down >/dev/null 2>&1
say ""
say "vLLM gestoppt. Ergebnisdateien: $(ls -1 "$OUT"/*.json 2>/dev/null | wc -l)"
say "Fertig um $(date)."
