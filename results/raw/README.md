# Rohdaten: welches Verzeichnis wofür

Kurzfassung: **`long_context_final/` ist der Datensatz, auf den sich alle
Ergebnistabellen der Thesis beziehen.** Die Verzeichnisnamen geben die
Reihenfolge nicht wieder, deshalb diese Übersicht.

`long_context_final/` enthält ein eigenes `MANIFEST` mit der vollständigen
Dateiliste und der Herkunft jeder einzelnen Datei.

## Verwendet

| Verzeichnis | Datum | Wofür |
|---|---|---|
| `long_context_final/` | 2026-07-17/18 | **Hauptdatensatz.** Speicher, Durchsatz, Perplexität, Retrieval. Gelesen von `scripts/verify_thesis_numbers.py` |
| `kv_distributions_v2/` | 2026-05-04 | Verteilungskennzahlen (Kurtosis, Outlier-Ratio, Dynamic Range, Varianzverhältnis) |
| `rotation/` | 2026-10-01 | Hadamard-Rotationsexperiment, in der Thesis als Pfad zitiert. Je Modell ein Lauf mit Perplexität und Kurtosis an WikiText-2, dazu die Kurtosis der zweiten Stichprobe (wiederholter Satz), beide 4096 Tokens. Gelesen von `scripts/statistical_tests.py` |
| `kv_split/` | 2026-08-04 | Getrennte Quantisierung von Keys und Values, als Pfad zitiert |
| `layerwise/` | 2026-08-04 | Layer-weise Quantisierung nach Kurtosis, als Pfad zitiert |
| `layer_exclusion/` | 2026-09-24 | Qwen2-7B mit je einem Layer in FP16, grenzt das INT4-Versagen auf Layer 0 ein (`scripts/experiment_layer_exclusion.py`) |
| `key_bias/` | 2026-09-24 | Bias der Key-Projektion in Layer 0 von Qwen2-7B (INT4, INT2, KIVI), Ursache des Qwen2-Versagens (`scripts/experiment_key_bias.py`) |
| `kv_tails/` | 2026-09-24 | Verteilungsränder der Keys aller fünf Modelle, gleiche Stichprobe wie `kv_distributions_v2/` (`analyze_kv_distributions.py --tail`) |

## Als Nachweis gebraucht, nicht für Ergebniswerte

| Verzeichnis | Datum | Wofür |
|---|---|---|
| `long_context_v2/` | 2026-07-17/18 | Der Messlauf, aus dem `long_context_final` hervorgeht. Enthält `kernel_probe_validation.json`, das die Thesis in einer Fußnote zitiert. Seine Gemma-Dateien zeigen den Standard-Kernel und tragen die Diagnosewerte der Gemma-Fallstudie |
| `long_context_v2_gemma_repeatkv/` | 2026-07-18 | Gemma-Wiederholung mit `--sdpa-force-repeat-kv`. Diese sechs Dateien sind die Gemma-Werte in `long_context_final` |
| `long_context/` | 2026-05-03/04 | Erster Messlauf unter dem alten Protokoll (volle Logits, Cache-Kopien im Peak). Belegt zusammen mit `long_context_v2` die Reproduzierbarkeit (Abschnitt 3.8 der Thesis) und dokumentiert den Protokolleffekt |
| `protocol_check/` | 2026-07-17 | Validierung der Protokolländerung vom 2026-07-17 |

## Abgelöst

| Verzeichnis | Datum | Abgelöst durch |
|---|---|---|
| `kv_distributions/` | 2026-05-03 | `kv_distributions_v2/` |

## Aus einer anderen Arbeit

| Verzeichnis | Datum | Wofür |
|---|---|---|
| `wissem_backend_comparison/` | 2026-05-18/19 | Messungen der Vorarbeit zu Attention-Backends. Gehört nicht zu dieser Thesis |

## Wie `long_context_final` entsteht

`long_context_final` ist kein eigener Messlauf, sondern eine Zusammenstellung:

- 24 Dateien byte-identisch aus `long_context_v2/` (Mistral, Yi, Qwen2, Qwen3)
- 6 Dateien aus `long_context_v2_gemma_repeatkv/` (Gemma)
- Die Gemma-Dateien aus `long_context_v2/` sind bewusst **nicht** enthalten:
  Dort lief Gemma auf dem Math-Fallback und brach oberhalb von 8k ab. Sie
  werden allein für die Fallstudie in Kapitel 4 herangezogen.

## Prüfung

`scripts/verify_thesis_numbers.py` vergleicht jede Zelle der Ergebnistabellen
gegen `long_context_final/` und `kv_distributions_v2/` und meldet zusätzlich,
wenn ein Verzeichnis hier nicht aufgeführt ist.
