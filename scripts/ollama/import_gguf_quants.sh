#!/usr/bin/env bash
# Holt die Gewichts-Quantisierungsleiter von HuggingFace und registriert sie in Ollama.
#
# Hintergrund: Die Ollama-Library fuehrt fuer qwen3.8:27b nur q4_K_M, q8_0 und
# bf16 -- q5_K_M und q6_K fehlen dort, also genau die Mitte der Leiter, in der
# der Sweetspot vermutet wird. unsloth/Qwen3.8-27B-GGUF hat die vollstaendige
# Leiter.
#
# WICHTIG -- Rezept-Konsistenz: unsloth liefert "UD"-Quants (Unsloth Dynamic,
# imatrix-kalibriert mit layer-weiser Bitverteilung). Die sind NICHT dasselbe
# Verfahren wie Ollamas Standard-q4_K_M; sichtbar an der Groesse (UD-Q4_K_M
# 15.33 GiB vs. Ollama q4_K_M 16.52 GiB). Auf einer Vergleichsachse duerfen die
# Rezepte nicht gemischt werden, sonst misst man das Quantisierungsverfahren
# statt der Bitbreite. Die Leiter kommt deshalb geschlossen von unsloth; das
# Ollama-Tag qwen3.8:27b laeuft separat als Referenzpunkt auf den Produktivstand.
#
#   ./import_gguf_quants.sh              # Standardleiter Q4_K_M/Q5_K_M/Q6_K
#   ./import_gguf_quants.sh --with-q8    # zusaetzlich Q8_0 (27 GiB, Qualitaetsdecke)
#   ./import_gguf_quants.sh --list       # nur zeigen, was geholt wuerde

set -euo pipefail

REPO="unsloth/Qwen3.8-27B-GGUF"
# Ablage im geteilten Ollama-Modellverzeichnis, damit der Container drankommt.
GGUF_DIR="//wsl.localhost/Ubuntu/home/crack/ai-models/ollama-models/gguf-import"
CONTAINER="ollama-bench"

# Datei -> Ollama-Tag. Die Tags tragen "ud" im Namen, damit sie im Ergebnis
# nicht mit dem Library-Quant verwechselt werden.
declare -a FILES=(
  "Qwen3.8-27B-UD-Q4_K_M.gguf|qwen3.8-ud:27b-q4_K_M"
  "Qwen3.8-27B-UD-Q5_K_M.gguf|qwen3.8-ud:27b-q5_K_M"
  "Qwen3.8-27B-UD-Q6_K.gguf|qwen3.8-ud:27b-q6_K"
)

LIST_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-q8) FILES+=("Qwen3.8-27B-Q8_0.gguf|qwen3.8-ud:27b-q8_0") ;;
    --list) LIST_ONLY=1 ;;
    *) echo "Unbekannte Option: $1" >&2; exit 2 ;;
  esac
  shift
done

echo "Repo:      $REPO"
echo "Ablage:    $GGUF_DIR"
echo "Container: $CONTAINER"
echo

if [[ $LIST_ONLY -eq 1 ]]; then
  for entry in "${FILES[@]}"; do
    echo "  ${entry%%|*}  ->  ${entry##*|}"
  done
  exit 0
fi

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "ABBRUCH: Container $CONTAINER laeuft nicht." >&2
  echo "         docker compose -f compose.bench.yml up -d" >&2
  exit 1
fi

mkdir -p "$GGUF_DIR"

for entry in "${FILES[@]}"; do
  file="${entry%%|*}"
  tag="${entry##*|}"
  target="$GGUF_DIR/$file"

  if [[ -f "$target" ]]; then
    echo "== $file bereits vorhanden ($(du -h "$target" | cut -f1))"
  else
    echo "== Lade $file ..."
    # -C - setzt abgebrochene Downloads fort; die Dateien sind 15-27 GiB gross.
    curl -L -C - --fail --progress-bar \
      -o "$target" \
      "https://huggingface.co/$REPO/resolve/main/$file"
  fi

  echo "== Registriere als $tag ..."
  # Modelfile im Container erzeugen und Modell anlegen. num_ctx wird bewusst
  # NICHT gepinnt: die Harness setzt es je Anfrage ueber options.num_ctx.
  docker exec "$CONTAINER" sh -c \
    "printf 'FROM /root/.ollama/gguf-import/%s\nPARAMETER temperature 0.2\n' '$file' > /tmp/Mf.$$ && ollama create '$tag' -f /tmp/Mf.$$ && rm -f /tmp/Mf.$$"
  echo
done

echo "Fertig. Registrierte Tags:"
docker exec "$CONTAINER" ollama list | grep -E "qwen3.8-ud|NAME" || true
