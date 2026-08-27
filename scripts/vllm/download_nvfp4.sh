#!/usr/bin/env bash
# Holt den NVFP4-Checkpoint fuer die vLLM-Gegenmessung.
#
# Zweck: Auf Blackwell liegen die FP8- und FP4-Tensorkerne brach, solange
# llama.cpp die GGUF-Integer-Quants zu FP16 dequantisiert und darin rechnet.
# vLLM kann NVFP4 nativ. Die Frage ist ausschliesslich, wie viel Durchsatz das
# bei ~128k Kontext bringt -- 55 -> 90 tok/s waere ein Grund fuer ein zweites
# Profil, 55 -> 65 nicht.
#
# Wichtig: NICHT waehrend einer laufenden Messung starten. Ein paralleler
# Download hat den Decode-Durchsatz messbar gedrueckt (112 -> 21 tok/s).
#
# Laeuft innerhalb von WSL, nicht ueber den //wsl.localhost-Mount -- bei 22 GiB
# kostet das 9P-Protokoll erheblich Zeit.
#
#   MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -- bash /mnt/d/.../download_nvfp4.sh

set -u

DEST="${NVFP4_DEST:-/home/crack/ai-models/vllm-models/Qwen3.8-27B-NVFP4}"
REPO="${NVFP4_REPO:-unsloth/Qwen3.8-27B-NVFP4}"

# model_mtp.safetensors ist der Draft-Kopf und liegt separat -- ohne ihn kein
# spekulatives Decoding, und genau darauf beruht der erhoffte Vorsprung.
FILES=(
  "config.json"
  "generation_config.json"
  "model.safetensors"
  "model.safetensors.index.json"
  "model_mtp.safetensors"
  "preprocessor_config.json"
  "tokenizer.json"
  "tokenizer_config.json"
  "vocab.json"
  "chat_template.jinja"
  "video_preprocessor_config.json"
)

mkdir -p "$DEST"
cd "$DEST" || exit 1

echo "Ziel: $DEST"
echo "Repo: $REPO"
df -h . | tail -1
echo

fail=0
for f in "${FILES[@]}"; do
  url="https://huggingface.co/$REPO/resolve/main/$f"
  remote=$(curl -sIL "$url" | grep -i '^content-length' | tail -1 | tr -d '\r' | awk '{print $2}')

  if [[ -f "$f" && -n "${remote:-}" ]]; then
    local_sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [[ "$local_sz" == "$remote" ]]; then
      echo "= $f  bereits vollstaendig ($(numfmt --to=iec "$local_sz"))"
      continue
    fi
  fi

  echo "= $(date +%H:%M:%S)  $f"
  if curl -L -C - --fail --retry 5 --retry-delay 10 --silent --show-error -o "$f" "$url"; then
    echo "    ok  $(numfmt --to=iec "$(stat -c %s "$f")")"
  else
    echo "    FEHLER bei $f"
    fail=$((fail + 1))
  fi
done

echo
echo "=== $(date +%H:%M:%S)  fertig, $fail Fehler"
du -sh "$DEST"
exit "$fail"
