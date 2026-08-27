#!/usr/bin/env bash
# Holt einen HuggingFace-Checkpoint fuer die vLLM-Gegenmessung.
#
# Ersetzt download_nvfp4.sh, das die Dateiliste noch fest verdrahtet hatte. Die
# Liste kommt jetzt aus der HF-API, damit beliebige Quants ohne Codeaenderung
# ladbar sind -- wir vergleichen mehrere Quantisierungsverfahren unter demselben
# Runtime.
#
# Wichtig: NICHT waehrend einer laufenden Messung starten. Ein paralleler
# Download hat den Decode-Durchsatz messbar gedrueckt (112 -> 21 tok/s).
#
# Laeuft innerhalb von WSL, nicht ueber den //wsl.localhost-Mount -- bei ~20 GiB
# kostet das 9P-Protokoll erheblich Zeit.
#
#   MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -- bash /mnt/d/.../download_checkpoint.sh \
#       RedHatAI/Qwen3.8-27B-INT4

set -u

REPO="${1:?Repo angeben, z.B. RedHatAI/Qwen3.8-27B-INT4}"
NAME="${REPO##*/}"
DEST="${CKPT_DEST:-/home/crack/ai-models/vllm-models}/$NAME"

mkdir -p "$DEST"
cd "$DEST" || exit 1

echo "Repo: $REPO"
echo "Ziel: $DEST"
df -h . | tail -1
echo

# Dateiliste aus der API. Ausgeschlossen wird nur, was fuer die Inferenz
# nachweislich nicht gebraucht wird -- lieber eine Datei zu viel als ein
# fehlender Tokenizer.
mapfile -t FILES < <(curl -s "https://huggingface.co/api/models/$REPO" \
  | python3 -c "
import json,sys
d=json.load(sys.stdin)
skip=('.gitattributes','README.md','LICENSE')
for s in d.get('siblings',[]):
    f=s.get('rfilename','')
    if not f or f.startswith('.') or f in skip: continue
    if '/' in f and not f.startswith('mtp'): continue   # Unterordner meiden
    print(f)
")

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "ABBRUCH: keine Dateiliste erhalten." >&2
  exit 1
fi
echo "${#FILES[@]} Dateien"

fail=0
for f in "${FILES[@]}"; do
  url="https://huggingface.co/$REPO/resolve/main/$f"
  remote=$(curl -sIL "$url" | grep -i '^content-length' | tail -1 | tr -d '\r' | awk '{print $2}')

  if [[ -f "$f" && -n "${remote:-}" ]]; then
    local_sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [[ "$local_sz" == "$remote" ]]; then
      echo "= $f  vollstaendig ($(numfmt --to=iec "$local_sz"))"
      continue
    fi
  fi

  echo "= $(date +%H:%M:%S)  $f"
  mkdir -p "$(dirname "$f")"
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
