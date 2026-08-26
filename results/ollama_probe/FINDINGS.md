# Zwischenstand: VRAM-Budget und Allokationsverhalten

Stand 2026-08-26. Interne Messung, **nicht Teil der Masterarbeit**.
Hardware: RTX 5090 (32607 MiB), Treiber 591.74.

Diese Datei hält fest, was vor dem eigentlichen Qualitätsdurchlauf schon
feststeht. Die Qualitätsachse (Needle-Retrieval, Perplexität) und der Durchsatz
über die Gewichts-Quant-Leiter fehlen noch — ohne die ist **kein Arbeitspunkt
bestimmbar**, weil Headroom allein nichts über die Ausgabequalität sagt.

## 1. Ollama allokiert den Kontext vollständig vorab

Sonde: dasselbe Modell, 6-Token-Prompt, nur `num_ctx` variiert. Würde nur der
tatsächliche Bedarf allokiert, dürfte die Kontextlänge kaum durchschlagen.

| `num_ctx` | Ollama 0.32.14 | Ollama 0.33.0 |
|---:|---:|---:|
| 8192 | 18569 MB | 18483 MB |
| 32768 | 19672 MB | 19607 MB |
| 131072 | 23578 MB | 23566 MB |

Die Deltas decken sich mit der analytischen KV-Größe (34 KiB/Token bei q8_0).
**Der gesamte Kontext wird vorab belegt, unabhängig von der Promptlänge.**

**Ein Upgrade löst das nicht.** 0.33.0 verhält sich innerhalb des Rauschens
identisch zu 0.32.14. Die Release Notes bis 0.33.0 nennen keine Änderung an der
Allokation; 0.32.15 bringt Metadaten-Caching (TTFT etwa halbiert), was für sich
genommen ein Grund zum Upgrade ist, aber keinen Speicher spart.

## 2. Belegung nach KV-Cache-Typ

Gemessen als Delta gegen den entladenen Zustand, also unabhängig vom
Desktop-Overhead. Gewichte: Library-`q4_K_M` (16,52 GiB), Ollama 0.33.0,
`OLLAMA_NUM_PARALLEL=1`, Flash-Attention an.

| KV-Typ | 128k | 192k | 256k |
|---|---:|---:|---:|
| `f16` | 26895 MB | 28492 MB | 30760 MB |
| `q8_0` *(aktuell produktiv)* | 23583 MB | 25872 MB | 27164 MB |
| `q4_0` | 21522 MB | 23318 MB | 25243 MB |

`q4_0` statt `q8_0` spart bei 128k rund **2,0 GB**, bei 256k rund **1,9 GB**.

## 3. Der native 256k-Kontext passt bereits heute

`docs/INFRASTRUCTURE.md` hält fest, dass 262144 die Karte zum OOM brachte —
das war allerdings unter `OLLAMA_NUM_PARALLEL=4`. Bei `P=1`, wie heute
konfiguriert, passt der volle native Kontext:

| Desktop-Overhead | q8_0 @256k | q4_0 @256k |
|---|---:|---:|
| minimal (1114 MB) | 28278 MB → 4329 MB frei | 26357 MB → 6250 MB frei |
| stark (4414 MB) | 31578 MB → 1029 MB frei | 29657 MB → 2950 MB frei |

Die 128k-Pinnung lässt also Reserve liegen. **Belastbar ist das aber erst,
wenn die Qualität bei 256k gemessen ist** — ein Kontextfenster, das ins VRAM
passt, ist noch keines, in dem das Modell zuverlässig arbeitet.

## 4. Der Desktop kostet bis zu 3 GB

Der Leerlaufwert der Karte schwankte im Verlauf der Messung zwischen 1114 MB
und 4414 MB, je nach geöffneten Fenstern (Explorer, SearchHost, NVIDIA Overlay,
EdgeWebView, TextInputHost). Das ist derselbe Betrag, um den `q4_0` gegenüber
`q8_0` entlastet — mit dem Unterschied, dass er nichts kostet.

## 5. Die Gewichte sind der größere Posten

Bei 128k entfallen ~17,7 GB auf die Gewichte und ~4,3 GB auf den KV-Cache.
Ursache ist die Architektur: `qwen35` ist hybrid, `full_attention_interval = 4`
bei 65 Blöcken, also tragen **nur ~16 Layer überhaupt KV**, die übrigen ~49 sind
SSM-Layer mit konstantem State. Der Hebel der KV-Quantisierung ist hier
strukturell kleiner als bei Dense-Modellen.

Deshalb ist die Gewichtsachse interessant: `unsloth/Qwen3.8-27B-GGUF` führt
`UD-Q4_K_M` mit 15,33 GiB gegenüber 16,52 GiB der Library bei nominell gleicher
Bitbreite — rund **1,2 GB weniger bei imatrix-Kalibrierung**. Ob das die
Qualität hält, ist offen und Teil des Durchlaufs.

## 6. Baseline-Durchsatz (Produktivkonfiguration)

Library-`q4_K_M` + `q8_0`-KV, Ollama 0.33.0. Puffergroessen aus dem
llama.cpp-Log, nicht aus `nvidia-smi`.

| Kontext | Prefill | Decode | KV-Puffer | Needle |
|---:|---:|---:|---:|---:|
| 8192 | 2998 tok/s | 112 tok/s | 272 MiB | 3/3 |
| 131072 | 1538 tok/s | 57 tok/s | 4352 MiB | 3/3 |

Der KV-Puffer entspricht in beiden Faellen exakt 34,0 KiB/Token. Decode halbiert
sich zwischen 8k und 128k -- fuer die Bewertung des Arbeitspunkts relevant, weil
Kontextlaenge nicht nur Speicher, sondern auch Geschwindigkeit kostet.

## Methodische Anmerkungen

**Speichermessung.** Primaerquelle sind die von llama.cpp gemeldeten Puffer
(Gewichte, KV, SSM-State, Compute) und nicht `nvidia-smi`. Geraetweites
Sampling enthaelt den Desktop-Overhead und ist damit die naive Variante; es
bleibt nur als Gegenprobe erhalten (Differenz zum Log ~630 MB, im Wesentlichen
der CUDA-Kontext). Ollamas eigenes `/api/ps` meldete 17044 statt 22952 MiB und
unterschlaegt Compute-Puffer und CLIP -- als Quelle unbrauchbar.

**Zwei korrigierte Messfehler.** Der Needle-Test zaehlte anfangs 0/5, weil
qwen3.8 als Thinking-Modell das Token-Budget im Feld `thinking` verbraucht und
`response` leer blieb; er laeuft jetzt mit `think:false`. Der Prefill-Durchsatz
wurde um Groessenordnungen zu hoch ausgewiesen (213920 tok/s bei 128k), weil die
Messlaeufe den Prompt-Cache des Warmups trafen: Ollama meldet dann die volle
`prompt_eval_count`, aber nur die Dauer des ungecachten Rests. Jeder Lauf nutzt
jetzt einen eindeutigen Prompt-Praefix.

## Offen

- Needle-Retrieval und Perplexität über die Matrix (Gewicht × KV × Kontext)
- Prefill-/Decode-Durchsatz je Zelle
- Ob `q4_0`-KV die Qualität hält — bei einem Qwen-Modell nicht selbstverständlich
- Verhalten bei 256k, jenseits des bisher gefahrenen 128k-Fensters
