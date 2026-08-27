# Arbeitspunkt qwen3.8-27b auf der RTX 5090

**Stand:** 2026-08-27, 12:00 · **Branch:** `experiments/qwen3.8-27b-ollama`
**Nicht Teil der Masterarbeit** — interne Untersuchung zum produktiven Betrieb.

**Leitfrage:** Welche Kombination aus Gewichts- und KV-Quantisierung liefert das beste
Verhältnis aus Ausgabequalität, Generierungsgeschwindigkeit und nutzbarer Kontextlänge?
Priorität laut Vorgabe: Qualität und tok/s vor Headroom — lieber häufiger compacten als
ein langes Fenster mit schlechterer Ausgabe.

---

## 1. Ergebnisse

Alle Zellen validiert (`validate_runs.py`): der von llama.cpp gemeldete KV-Puffer stimmt
mit dem ausgewiesenen KV-Typ überein. VRAM aus den llama.cpp-Puffern, nicht aus `nvidia-smi`.

| Kontext | Gewichte | KV | VRAM | davon KV | Prefill | Decode | n |
|---:|---|---|---:|---:|---:|---:|---:|
| 8 192 | Library | `f16` | 18 300 MiB | 512 MiB | 2 263 | **105,2** | 2 |
| 8 192 | Library | `q4_0` | 17 932 MiB | 144 MiB | 2 962 | **109,0** | 2 |
| 8 192 | Library | `q8_0` | 18 060 MiB | 272 MiB | 2 639 | **111,6** | 2 |
| 32 768 | Library | `f16` | 19 980 MiB | 2 048 MiB | 2 801 | **114,3** | 2 |
| 32 768 | Library | `q4_0` | 18 552 MiB | 576 MiB | 2 820 | **113,4** | 2 |
| 32 768 | Library | `q8_0` | 19 064 MiB | 1 088 MiB | 2 855 | **109,7** | 5 |
| 32 768 | UD+Vision | `q8_0` | 18 730 MiB | 1 088 MiB | 2 875 | **115,5** | 5 |
| 131 072 | Library | `f16` | 26 322 MiB | 8 192 MiB | 598 | **61,7** | 2 |
| 131 072 | Library | `q8_0` | 22 952 MiB | 4 352 MiB | 1 570 | **80,2** | 5 |
| 131 072 | UD+Vision | `q8_0` | 22 618 MiB | 4 352 MiB | 1 596 | **88,4** | 5 |
| 262 144 | Library | `q4_0` | 24 616 MiB | 4 608 MiB | 991 | **55,2** | 3 |

*Library* = `qwen3.8:27b` (Ollama-Library, Q4_K_M, 15,65 GiB Gewichte).
*UD+Vision* = `qwen3.8-udv:27b-q4_K_M` (unsloth Dynamic, mit Projektor und MTP-Draft,
15,33 GiB). Beide mit `RENDERER qwen3.8`, `PARSER qwen3.5`, `draft_num_predict 4`.

**Qualität: in allen 13 Zellen makellos.** Needle 5/5 über fünf Tiefen, Multi-Needle 5/5,
Verbatim-Codeblock exakt reproduziert — bei jedem Gewicht, jedem KV-Typ und jeder
Kontextlänge bis 260 809 Token. Es gibt bislang **keinen einzigen Messpunkt mit
messbarem Qualitätsverlust.**

### Was daraus folgt

**Die KV-Quantisierung ist qualitativ kostenlos.** Selbst `q4_0` bei 260k Token
reproduziert einen Codeblock zeichengenau. Die Sorge, ein Qwen-Modell könnte bei
aggressiver KV-Quantisierung einbrechen, bestätigt sich auf dem llama.cpp-Pfad nicht.
*(Einschränkung: die Sonden messen Retrieval und Reproduktion, nicht Perplexität.
Schleichende Verschlechterung in freier Generierung wäre damit nicht zwingend sichtbar —
siehe offener Punkt O6.)*

**Unterhalb von 128k entscheidet der KV-Typ praktisch nichts.** Bei 8k und 32k liegen
`f16`, `q8_0` und `q4_0` alle zwischen 105 und 117 tok/s. Erst bei 128k greift die
Speicherbandbreite: `f16` bricht auf 61,7 ein, `q8_0` hält 80,2. Wer unter 32k arbeitet,
kann den KV-Typ ignorieren.

**Der Gewichts-Quant ist der falsche Hebel.** Der Unterschied zwischen Library-Q4_K_M und
unsloth-UD beträgt real **327 MiB** an Gewichten (15 339 gegen 15 009 MiB auf der GPU),
nicht die 1,19 GiB, die der Dateigrößenvergleich nahelegt — Ollamas 16,52 GiB umfassen
Modell *plus* Projektor. 334 MiB entsprechen bei `q8_0` rund 10k zusätzlichen Token. Für
einen Wechsel von einem Library-Tag auf einen selbst gepflegten Import zu wenig.

Der UD-Quant ist allerdings **konsistent 5–8 tok/s schneller** bei gleicher Qualität
(115,5 gegen 109,7 bei 32k; 88,4 gegen 80,2 bei 128k). Das ist der eigentliche Vorteil,
nicht der Speicher.

**Der native 256k-Kontext funktioniert.** Mit `q4_0` passt er in 24,6 GB und liefert
55,2 tok/s bei unveränderter Qualität. Der in `INFRASTRUCTURE.md` dokumentierte OOM trat
unter `OLLAMA_NUM_PARALLEL=4` auf; bei `P=1` ist er kein Thema.

### Der Tradeoff in einer Zeile

| | heute (128k, `q8_0`) | 256k, `q4_0` | Δ |
|---|---:|---:|---:|
| Kontext | 131 072 | 260 809 | **+99 %** |
| Decode | 80,2 tok/s | 55,2 tok/s | **−31 %** |
| Prefill | 1 570 tok/s | 991 tok/s | −37 % |
| VRAM | 22 952 MiB | 24 616 MiB | +1 664 MiB |
| Qualität | 1,000 | 1,000 | ±0 |

**Der Prefill ist dabei das größere Problem als der Decode.** Ein volles Fenster zu füllen
kostet bei 128k rund 83 Sekunden, bei 256k **4,4 Minuten** — jedes Mal, wenn der
Prompt-Cache nicht greift. Und weil Ollama den Kontext vorab allokiert, bezahlt man die
24,6 GB permanent, auch bei kurzen Prompts.

---

## 2. VRAM-Budget: der Embedder fehlt in obiger Rechnung

Die Tabellenwerte sind reine Chat-Modell-Zahlen. Produktiv läuft `qwen3-embedding:4b-ctx2k`
daneben (Qdrant-Ingest, LiteLLM-Pool `embed-local`), laut `compose.yaml` ~3,8 GB.

| Posten | 128k / `q8_0` | 256k / `q4_0` |
|---|---:|---:|
| Chat-Modell *(gemessen)* | 22 952 MiB | 24 616 MiB |
| Embedder 4b *(aus Doku, **ungemessen**)* | ~3 800 MiB | ~3 800 MiB |
| Windows-Desktop *(gemessen, schwankend)* | 1 114–4 414 MiB | 1 114–4 414 MiB |
| **Summe** | **27 866–31 166** | **29 530–32 830** |
| von 32 607 MiB | passt | **oberes Ende passt nicht** |

Bei ruhigem Desktop bleiben für 256k gut 3 GB Luft; mit offenem Browser und
NVIDIA-Overlay reicht es nicht. **Der Embedder ist der Posten, der 256k unter Windows von
„geht" auf „wackelig" schiebt.** Die 3,8 GB sind aus eurer Doku übernommen und noch nicht
nachgemessen (offener Punkt O3).

Der Desktop-Overhead schwankte im Verlauf der Messungen zwischen 1 114 MiB (alles
geschlossen) und 4 414 MiB (Explorer, SearchHost, StartMenu, NVIDIA Overlay,
EdgeWebView, TextInputHost). Das ist derselbe Betrag, den `q4_0` gegenüber `q8_0`
einspart — nur kostenlos.

---

## 3. Ollama allokiert den Kontext vollständig vorab

Sonde: identisches Modell, 6-Token-Prompt, nur `num_ctx` variiert.

| `num_ctx` | Ollama 0.32.14 | Ollama 0.33.0 |
|---:|---:|---:|
| 8 192 | 18 569 MB | 18 483 MB |
| 32 768 | 19 672 MB | 19 607 MB |
| 131 072 | 23 578 MB | 23 566 MB |

**Ein Upgrade löst das nicht** — 0.33.0 verhält sich innerhalb des Rauschens identisch.
Die Release Notes bis 0.33.0 nennen keine Änderung an der Allokation. 0.32.15 bringt
Metadaten-Caching mit etwa halbierter TTFT; das ist ein eigener Upgrade-Grund, spart aber
keinen Speicher. **Ihr fahrt aktuell 0.32.14.**

---

## 4. Laufzeiten (für die Planung des nächsten Durchgangs)

Gemessen, saubere Werte von vor der Überlappung (siehe Abschnitt 7).

| Zelle | Kontexte | Messläufe | Dauer |
|---|---|---:|---:|
| Library, `f16` | 8k/32k/128k | 2 | 52 min |
| Library, `q8_0` | 8k/32k/128k | 2 | 24 min |
| Library, `q4_0` | 8k/32k/128k | 2 | 16 min |
| UD-Q4_K_M, `q8_0` | 8k/32k/128k | 2 | 19 min |
| UD-Q5_K_M, `q8_0` | 8k/32k/128k | 2 | 20 min |
| UD-Q6_K, `q8_0` | 8k/32k/128k | 2 | 21 min |
| UD-Q6_K, `f16` | 8k/32k/128k | 2 | 69 min |
| Library, `q8_0` | 32k/128k | 5 | 42 min |
| Library, `q4_0` | 256k | 3 | ~83 min *(mit Störung)* |

**Faustformel.** Eine Zelle fährt rund **12 volle Prefills** (1 Warmup + n Messläufe +
5 Needle-Tiefen + 4 Qualitätssonden). Die Dauer ergibt sich fast vollständig daraus:

```
Zelldauer ≈ 12 × (Kontextlänge / Prefill-tok/s) + ~2 min Modell-Load
```

| Kontext | Prefill-Rate | ein Prefill | ≈ Zelldauer |
|---:|---:|---:|---:|
| 8 192 | ~2 800 tok/s | 3 s | ~3 min |
| 32 768 | ~2 850 tok/s | 12 s | ~4 min |
| 131 072 | ~1 570 tok/s | 83 s | **~19 min** |
| 262 144 | ~991 tok/s | 264 s | **~55 min** |

Große Modelle mit `f16`-KV sind überproportional langsam (69 min für UD-Q6_K), weil sie
nahe an die VRAM-Grenze kommen — dort bricht der Prefill ein (bis auf 365 tok/s gemessen).

**Weitere Zeitposten:** GGUF-Download 55 GiB ≈ 10 min (~75 MB/s, in WSL; über den
`//wsl.localhost`-Mount deutlich langsamer). `ollama create` je Modell ≈ 3 min (kopiert
den Blob). Docker-Image-Pull Ollama ≈ 2 min. Container-Neustart je KV-Wechsel ≈ 20 s.

---

## 5. Offene Punkte und Testplan

Nach Nutzen sortiert, mit geschätzter Laufzeit aus der Faustformel oben.

### Priorität 1 — schließt die Kurve, auf der der Arbeitspunkt liegt

**O1 · `q4_0` bei 131 072** — *fehlt* (Zelle beim Container-Neustart des Parallellaufs
gestorben). Ohne sie lässt sich nicht trennen, was der längere Kontext kostet und was der
KV-Typ. Direkter Vergleich gegen `q8_0` @128k = 80,2 tok/s.
→ **~20 min**

**O2 · Zwischenschritte 163 840 und 196 608** — zwischen 128k (80,2) und 256k (55,2)
klafft eine Lücke. Fällt der Decode linear oder gibt es eine Kante? Davon hängt ab, ob ein
Fenster von z. B. 192k der bessere Kompromiss ist als 256k.
→ **~60 min** für beide, mit `q4_0`

**O3 · Embedder-Fußabdruck messen** — die 3,8 GB stammen aus der Doku. Da an dieser Zahl
hängt, ob 256k produktiv überhaupt passt, sollte sie gemessen sein. Zusätzlich: bleibt der
Embedder mit `OLLAMA_KEEP_ALIVE=-1` dauerhaft resident oder wird er verdrängt?
→ **~10 min**

### Priorität 2 — billige Gewinne und Absicherung

**O4 · Desktop-Overhead minimieren** — was bringt es, NVIDIA Overlay und EdgeWebView zu
beenden bzw. die Anzeige auf die iGPU zu legen? Die 1,1–4,4 GB sind der größte einzelne
Hebel und kosten keine Qualität.
→ **~15 min**

**O5 · UD+Vision bei 256k** — läuft gerade. Zeigt, ob die 5–8 tok/s Vorsprung des
UD-Quants auch bei 256k bestehen.
→ *läuft*

**O6 · Perplexität als Qualitätsmaß** — die Sonden messen Retrieval und Reproduktion.
Beides war überall perfekt, was den Verdacht nährt, dass sie zu grob sind für schleichende
Verschlechterung. Perplexität bräuchte llama.cpp mit CUDA; der Image-Pull war beim ersten
Versuch nicht durchgekommen, und ob Upstream-llama.cpp die `qwen35`-Architektur kennt, ist
ungeprüft. Alternative ohne neues Tooling: die bereits eingesammelten Greedy-Ausgaben
gegen die `f16`-Referenz auswerten — dafür fehlen aktuell nur die `f16`-Gegenstücke bei
128k/256k.
→ **~30 min** für die `f16`-Referenzzellen, Auswertung dann rechnerisch

### Priorität 3 — größerer Aufwand, unsicherer Ertrag

**O7 · Nativ Ubuntu** — der dokumentierte Vorsprung von **+16 %** (181,3 gegen
156,5 tok/s) stammt aus einem Testprompt mit **leerem KV-Cache** (`"Zaehle von 1 bis 50."`,
22 Token). Ob er bei gefülltem 128k-Cache genauso ausfällt, ist offen: Decode ist dort
speicherbandbreiten- und nicht treiberlimitiert. Dazu entfällt der Desktop-Overhead.
Die Harness ist portabel; anzupassen sind nur Modellpfad und Compose-Datei.
→ **~3 h** für die vier wichtigsten Zellen, plus Boot und Einrichtung

**O8 · vLLM mit NVFP4** — siehe Abschnitt 6. Ergebnis der Vorrecherche: auf einer
32-GB-Karte tauscht man Kontext gegen Geschwindigkeit. Messenswert ist die eine Zahl, die
keine Doku liefert: tok/s bei ~128k auf dieser Karte.
→ **~2 h** plus 22 GiB Download

---

## 6. Recherche: vLLM und NVFP4

**vLLM unterstützt das Modell offiziell** (eigene Recipe-Seite, min. vLLM 0.17.0). Die
Architektur deckt sich mit dem, was der GGUF-Header hergibt: 64 Layer, davon 16 volle
Attention, 48 Gated-DeltaNet mit konstantem State. MTP-Draft funktioniert dort ebenfalls,
mit **Akzeptanzraten von 0,754–0,897** — deutlich über den 0,39–0,86, die hier im
Ollama-Log auftreten. Da steckt Tempo drin.

**Warum NVFP4 auf einer 5090 trotzdem Kontext kostet** — drei Gründe stapeln sich:

1. **NVFP4 ist nicht durchgehend 4 Bit.** Der unsloth-Checkpoint misst **21,81 GiB**
   gegenüber 15,33 GiB des GGUF — mixed precision aus `U8` (gepacktes FP4), `F8_E4M3` und
   `BF16`, effektiv ~6,4 Bit je Parameter statt 4. Echte 4 Bit wären 13,7 GiB.
   **6,5 GiB mehr Gewichte fehlen anschließend beim KV.**
2. **vLLMs KV-Untergrenze ist FP8** (~32 KiB/Token). llama.cpp fährt `q4_0` mit
   18 KiB/Token — vLLM braucht pro Token **1,8×** so viel.
3. **Der DeltaNet-State wird pro Sequenz-Slot allokiert.** Default `--max-num-seqs 256`.
   Deshalb steigt die Kapazität, wenn man ihn deckelt.

Aus der Recipe für eine einzelne 5090, KV-Token-Kapazität:

```
nur --enforce-eager        →  91 022 Token
+ --language-model-only    → 135 926   (ohne Vision)
+ --max-num-seqs 8         → 152 917
```

`--enforce-eager` ist nicht optional: ohne CUDA-Graphs zu deaktivieren stirbt der Start im
Graph-Capture mit OOM (nutzbar sind nur 31,4 GB). Das kostet für sich schon Durchsatz —
also ausgerechnet das, wofür man NVFP4 nimmt.

**Einordnung.** Da Vision beim Coding entbehrlich ist, wäre die realistische Zielmarke
`--language-model-only --max-num-seqs 8` mit ~150k Kapazität. Das entspricht ungefähr dem
heutigen 128k-Fenster; die 260k wären weg. Die Frage reduziert sich damit auf: **wie viel
schneller ist NVFP4 bei ~128k?** 55 → 90 tok/s wäre ein echter Gewinn, 55 → 65 nicht.

**Protokollfrage.** vLLM ist OpenAI-kompatibel; LiteLLM routet mit `openai/<modell>` +
`api_base` dorthin. Tool-Calling löst vLLM serverseitig über `--reasoning-parser qwen3`
und `--tool-call-parser qwen3_coder`. Der Aufwand liegt **nicht** im Backend-Tausch,
sondern in der Client-Migration: Cline (`~/.cline/…/providers.json` → `127.0.0.1:11434`),
OpenClaw (`ollama/qwen3.8:27b-ctx128k`), n8n (`OLLAMA_MODEL`) und die TrueNAS-App sprechen
heute alle **direkt Ollama-nativ**, nicht über LiteLLM. Diese Migration lohnt unabhängig
von vLLM — sie entkoppelt die Clients vom Backend und macht jeden künftigen Wechsel zur
Konfigurationsfrage.

Quellen: [vLLM Recipe Qwen3.8-27B](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) ·
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)

---

## 7. Messfallen (alle real aufgetreten)

Institutionelles Wissen — jede dieser Fallen hat plausibel aussehende, falsche Zahlen
produziert.

**Prompt-Cache verfälscht den Prefill.** Wiederholt ein Messlauf den Warmup-Prompt, meldet
Ollama die volle `prompt_eval_count`, aber nur die Dauer des ungecachten Rests. Gemessen:
213 920 tok/s bei 128k statt der realen 1 570. *Abhilfe:* eindeutiger Prompt-Präfix je Lauf.

**Thinking frisst das Antwortbudget.** qwen3.8 gibt Reasoning im separaten Feld `thinking`
aus. Bei `num_predict 24` bleibt `response` leer — der Needle-Test zählte 0/5, obwohl das
Modell korrekt arbeitete. *Abhilfe:* `think: false` für Retrieval-Sonden; für den Durchsatz
bleibt es an, weil tok/s über `eval_count` selbstnormalisierend ist.

**Zeitbasierte Log-Fenster addieren Ladevorgänge.** Bei zu weitem `docker logs --since`
summierte der Parser mehrere Model-Loads: 36 120 statt 18 060 MiB. *Abhilfe:* auf den
letzten `llama_model_loader:`-Block zuschneiden.

**`--kv-cache-type` ist nur eine Beschriftung.** Wirksam ist `OLLAMA_KV_CACHE_TYPE` im
Container. Am 2026-08-27 lief ein per `TaskStop` vermeintlich beendeter Lauf als
**Waisenprozess** weiter und konfigurierte den Container unter dem zweiten Lauf um — fünf
Zellen wiesen `q4_0` aus, liefen aber mit `q8_0`. In einer Datei wechselte der Typ sogar
zwischen den Kontexten. *Abhilfe:* `validate_runs.py` prüft die gemeldete KV-Puffergröße
gegen den ausgewiesenen Typ; `bench_ollama.py` tut dasselbe inline und warnt.
**Vor jedem Lauf prüfen, dass kein zweiter Messprozess läuft.**

**Ollamas `/api/ps` unterschlägt Speicher.** Meldete 17 044 statt 22 952 MiB — Compute-Puffer
und CLIP fehlen. Als Quelle unbrauchbar; die llama.cpp-Logzeilen sind maßgeblich.

**Gerätweites `nvidia-smi` enthält den Desktop.** Schwankte um bis zu 3 GB. Nur als
Gegenprobe verwendbar (Differenz zu den Puffersummen ≈ 630 MiB CUDA-Kontext).

**Unfairer Modellvergleich durch unvollständigen Import.** Die ersten UD-Modelle wurden mit
`FROM <gguf>` allein registriert — ohne Vision-Projektor, ohne `RENDERER`/`PARSER` und ohne
`draft_num_predict`, mit dem die Library-Variante spekulatives Decoding fährt. Ergebnis war
ein scheinbarer UD-Rückstand von 40–60 %, der nach korrekter Registrierung in einen
**Vorsprung** von 5–8 tok/s umschlug. Der MTP-Kopf steckt in beiden GGUFs (866 Tensoren,
identisch, inkl. `blk.64.nextn.*`) — er war nur nie eingeschaltet.
*Abhilfe:* Modelfile der Referenz mit `/api/show` auslesen und nachbilden.

---

## 8. Verzeichnisse

| Pfad | Inhalt |
|---|---|
| `raw/` | gültige Messläufe, von `validate_runs.py` geprüft |
| `raw_superseded/` | UD-Läufe ohne Draft/Vision — Qualität und KV-Größen gültig, Decode und VRAM-Vergleich nicht |
| `raw_mislabeled/` | Zellen mit falschem KV-Etikett aus dem Parallellauf — unbrauchbar |
| `*.log` | Laufprotokolle |
