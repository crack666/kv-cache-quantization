# Ollama/llama.cpp-Messpfad — Gewichts- und KV-Quantisierung

**Diese Skripte gehören nicht zur Masterarbeit.** Sie liegen auf dem Branch
`experiments/qwen3.8-27b-ollama` und dienen der internen Frage, welche
Kombination aus Gewichts- und KV-Cache-Quantisierung auf der RTX 5090 bei langem
Kontext den besten Kompromiss liefert. Nichts davon fließt in die Arbeit ein.

## Warum ein eigener Messpfad

Die Arbeit hält die Gewichte bewusst in FP16 und quantisiert ausschließlich den
Cache. Für qwen3.8-27b sind das rund 54 GB allein an Gewichten — mehr als die
32 GB der Karte. **Auf dieser Hardware ist die Methode der Arbeit auf dieses
Modell nicht anwendbar.** Die Gewichtsquantisierung ist hier deshalb keine
Erweiterung des Versuchsaufbaus, sondern dessen Voraussetzung.

Dazu kommt, dass der produktive Stack ein anderer ist: GGUF unter llama.cpp
statt safetensors unter HuggingFace, und llama.cpp-KV-Typen (`f16`/`q8_0`/`q4_0`)
statt `QuantizedCache` mit quanto/HQQ. Zwei verschiedene Quantisierer.

## Warum die Zahlen nicht direkt vergleichbar sind

Gegenüber der Arbeit unterscheiden sich **zwei** Dinge gleichzeitig: die
Gewichtsprazision und die KV-Implementierung. Ein direkter Zahlenvergleich wäre
deshalb nicht interpretierbar.

Dafür gibt es **Track B**: `qwen3:8b-fp16` ist exakt ein Modell des
Thesis-Testfelds (Qwen3-8B, Key-Kurtosis 23.01, dort fragil — bricht bei INT2
ein), gefahren in fp16 wie dort. Damit variiert nur noch die
KV-Implementierung. Erst wenn Track B zeigt, dass llama.cpp denselben Schaden
anrichtet wie quanto/HQQ, lässt sich Track A überhaupt gegen die Arbeit halten.

## Architektur des Messobjekts

qwen3.8-27b ist kein Dense-Transformer. Aus dem GGUF-Header:

| Kennwert | Wert |
|---|---|
| Architektur | `qwen35` (hybrid SSM + Attention) |
| Blöcke | 65 |
| `full_attention_interval` | 4 → **~16 Layer tragen KV**, ~49 sind SSM |
| Q-Heads / KV-Heads | 24 / 4 (GQA 6:1) |
| `key_length` / `value_length` | 256 / 256 (nicht die üblichen 128) |
| Nativer Kontext | 262144 |

Daraus folgt analytisch `2 × 4 × 256 × 16` = 64 KiB/Token bei f16, bei q8_0
~34 KiB/Token → **4,25 GiB @128k / 8,5 GiB @256k**. Das deckt sich mit den in
`ai-stack/docs/INFRASTRUCTURE.md` gemessenen Werten, die Rechnung trägt also.

Praktische Folge: Weil nur ein Viertel der Layer überhaupt KV hält, ist der
Hebel der KV-Quantisierung hier strukturell kleiner als bei Dense-Modellen —
das VRAM-Budget wird vom Gewichts-Quant dominiert. `bench_ollama.py` gibt den
analytischen Wert je Lauf mit aus, als Gegenprobe zum gemessenen VRAM-Peak.

## Isolation gegen den Produktivbetrieb

`OLLAMA_KV_CACHE_TYPE` ist eine Container-Env-Variable und lässt sich nicht je
Anfrage setzen — ein Durchlauf über drei KV-Stufen bräuchte also drei Neustarts.
Auf der Produktionsinstanz (Port 11434) wäre das ein Eingriff in einen
laufenden Dienst; OpenClaw, Cline, n8n und LiteLLM hängen daran.

Deshalb läuft die Messung auf einer **eigenen Instanz auf Port 11435**
(`compose.bench.yml`, Projektname `kvq-bench`) mit eigenem Env-Satz. Das
Modellverzeichnis wird geteilt, damit die 15–27 GiB großen Blobs nicht doppelt
auf der Platte liegen.

`bench_ollama.py` bricht ab, wenn als Host `:11434` angegeben wird.
`run_matrix.sh` bricht ab, wenn `ollama-container` noch läuft — beide Instanzen
gleichzeitig würden sich die 32 GB VRAM teilen und die Messung wertlos machen.

## Quant-Leiter: Rezepte nicht mischen

Die Ollama-Library führt für qwen3.8:27b nur `q4_K_M`, `q8_0` und `bf16` — genau
die Mitte der Leiter fehlt. Sie kommt deshalb von `unsloth/Qwen3.8-27B-GGUF`.

**Wichtig:** unsloth liefert *UD*-Quants (Unsloth Dynamic, imatrix-kalibriert
mit layer-weiser Bitverteilung). Das ist nicht dasselbe Verfahren wie Ollamas
Standard-`q4_K_M`; sichtbar an der Größe (UD-Q4_K_M 15,33 GiB vs. Library
16,52 GiB). Auf einer Vergleichsachse dürfen die Rezepte nicht gemischt werden,
sonst misst man das Quantisierungsverfahren statt der Bitbreite.

Die Achse kommt daher geschlossen von unsloth (`qwen3.8-ud:*`); das Library-Tag
`qwen3.8:27b` läuft separat als **Referenzpunkt auf den Produktivstand** mit.

## Ablauf

```bash
cd scripts/ollama
cp .env.example .env          # Pfad ggf. anpassen

docker stop ollama-container  # Produktion muss aus sein
docker compose -f compose.bench.yml up -d

./import_gguf_quants.sh       # ~54 GiB von HuggingFace, einmalig
./run_matrix.sh               # Track A
./run_matrix.sh --track-b     # Brücke zur Arbeit

docker compose -f compose.bench.yml down
docker start ollama-container # Produktion zurück
```

`./run_matrix.sh --dry-run` zeigt vorher, was liefe.

## Messprotokoll

Pro Zelle (Gewichts-Quant × KV-Typ × Kontextlänge):

- **Prefill/Decode-Durchsatz** getrennt, aus Ollamas eigenen Timings
  (`prompt_eval_*` / `eval_*`), Median über *n* Läufe
- **VRAM-Peak** per `nvidia-smi`-Sampling ab dem *entladenen* Zustand, damit der
  Ladevorgang im Peak enthalten ist. Gerätweit gemessen, nicht prozessweise —
  für die Budgetfrage zählt die Belegung der Karte.
- **Needle-Retrieval** über fünf Tiefen (0.1/0.25/0.5/0.75/0.9), Protokoll wie
  in `profiler_suite.py`

**Thinking-Modus:** qwen3.8 ist ein Thinking-Modell und gibt Reasoning in einem
eigenen Feld `thinking` aus, das nicht in `response` landet. Beim Needle-Test
ist Thinking deshalb **abgeschaltet** — sonst verbraucht das Reasoning das
Token-Budget, `response` bleibt leer und das zählt fälschlich als
Retrieval-Fehler (genau so in der ersten Testfassung passiert). Für den
Durchsatz bleibt es **an**, weil das dem produktiven Betrieb entspricht und
tok/s über `eval_count` ohnehin selbstnormalisierend ist.

**Token-Längen:** Ollama stellt keinen Tokenizer-Endpunkt bereit. Die Harness
kalibriert Zeichen-je-Token einmalig über `prompt_eval_count` eines Probelaufs
und skaliert den Haystack darauf. Der tatsächlich erreichte Wert steht je
Messung als `prompt_tokens_measured` im Ergebnis.

**Fehler sind Messwerte:** OOM und Fensterüberschreitungen werden als Zelle mit
`error` protokolliert statt den Lauf abzubrechen — sie markieren die Grenze des
Budgets und gehören in die Pareto-Front.

## Ausgabe

JSON nach `results/ollama_probe/raw/`, Schema v2.0 wie
`results/raw/long_context_v2`, ergänzt um einen `runtime`-Block, der den
llama.cpp-Pfad kennzeichnet. Die Thesis-Verzeichnisse (`results/raw/`,
`results/figures/`, `results/tables/`) werden nicht angefasst.
