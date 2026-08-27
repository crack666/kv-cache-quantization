#!/usr/bin/env bash
# Laedt die GGUF-Quantisierungsleiter von HuggingFace.
#
# Laeuft bewusst INNERHALB von WSL und nicht ueber den //wsl.localhost-Mount:
# bei zusammen ~54 GiB kostet das 9P-Protokoll des UNC-Pfads erheblich Zeit.
#
#   wsl.exe -d Ubuntu -- bash /mnt/d/.../download_quants.sh
#
# Abgebrochene Downloads werden fortgesetzt (curl -C -), der Aufruf ist also
# gefahrlos wiederholbar.

set -u

DEST="${GGUF_DEST:-/home/crack/ai-models/ollama-models/gguf-import}"
REPO="${GGUF_REPO:-unsloth/Qwen3.8-27B-GGUF}"

FILES=(
  "Qwen3.8-27B-UD-Q4_K_M.gguf"
  "Qwen3.8-27B-UD-Q5_K_M.gguf"
  "Qwen3.8-27B-UD-Q6_K.gguf"
)

mkdir -p "$DEST"
cd "$DEST" || exit 1

echo "Ziel: $DEST"
echo "Repo: $REPO"
df -h . | tail -1
echo

fail=0
for f in "${FILES[@]}"; do
  echo "=== $(date +%H:%M:%S)  $f"

  # Erwartete Groesse vorab holen, um vollstaendige Dateien zu ueberspringen.
  remote=$(curl -sIL "https://huggingface.co/$REPO/resolve/main/$f" \
           | grep -i '^content-length' | tail -1 | tr -d '\r' | awk '{print $2}')
  if [[ -f "$f" && -n "${remote:-}" ]]; then
    local_sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [[ "$local_sz" == "$remote" ]]; then
      echo "    bereits vollstaendig ($(numfmt --to=iec "$local_sz"))"
      continue
    fi
    echo "    unvollstaendig ($(numfmt --to=iec "$local_sz") von $(numfmt --to=iec "$remote")), setze fort"
  fi

  if curl -L -C - --fail --retry 5 --retry-delay 10 --silent --show-error \
          -o "$f" "https://huggingface.co/$REPO/resolve/main/$f"; then
    echo "    ok  $(numfmt --to=iec "$(stat -c %s "$f")")"
  else
    echo "    FEHLER bei $f"
    fail=$((fail + 1))
  fi
done

echo
echo "=== $(date +%H:%M:%S)  fertig, $fail Fehler"
ls -la "$DEST"/*.gguf 2>/dev/null
df -h "$DEST" | tail -1
exit "$fail"
