# Arbeitspunkt qwen3.8-27b auf der RTX 5090

**Stand:** 2026-08-27, 15:20 · **Branch:** `experiments/qwen3.8-27b-ollama`
**Nicht Teil der Masterarbeit** — interne Untersuchung zum produktiven Betrieb.

**Leitfrage:** Welche Kombination aus Kontextlänge und KV-Quantisierung liefert das beste
Verhältnis aus Qualität, Geschwindigkeit und Speicherreserve? Priorität laut Vorgabe:
Qualität und tok/s vor Headroom — lieber häufiger compacten als ein langes Fenster mit
schlechterer Ausgabe. Randbedingung: Windows bleibt der Alltagsarbeitsplatz (die Maschine
wird auch zum Spielen genutzt), der Embedder läuft mit, und Jarvis braucht künftig
gleichzeitig VRAM.

---

## 1. Empfehlung

**Für den Alltag: `q4_0`-KV bei 131 072.** Gegenüber dem heutigen Stand (`q8_0`, 131 072)
kostet das 4 % Durchsatz und bringt **2 GB Reserve** — genau den Betrag, um den euer
Desktop schwankt. Damit hört die Konfiguration auf, auf Kante genäht zu sein.

**Wenn der lange Kontext gebraucht wird: 262 144, ebenfalls `q4_0`.** Das kostet bei
kurzer Session nur 4,5 % und funktioniert ohne Reload für alles darunter — aber es bleibt
nur **1 GB frei**, und leichtes Spilling ist messbar. Als Dauerkonfiguration nur, wenn
sonst nichts Größeres auf der Karte liegt.

**Als Kompromiss: 196 608.** 50 % mehr Fenster als heute, rund 2,4 GB Reserve.

Was in jedem Fall gilt: **`f16`-KV hat keinen Grund mehr.** Es kostet bei 131 072 fast
4 GB mehr als `q8_0` und ist mit 61,7 gegen 80,2 tok/s deutlich langsamer — die
Qualität war in keiner einzigen Zelle besser.

---

## 2. Der zentrale Befund: Allokation ist gratis, Füllstand kostet

Gemessen bei **fester Füllung**, nur `num_ctx` variiert (`q8_0`):

| Füllung | `num_ctx` | Decode | VRAM |
|---:|---:|---:|---:|
| 8 192 | 8 192 | 137,7 | 21 992 MiB |
| 8 192 | 32 768 | 138,5 | 22 959 MiB |
| 8 192 | 131 072 | 137,4 | 26 954 MiB |
| 8 192 | **262 144** | **62,5** | **30 976 MiB** |
| 32 768 | 32 768 | 131,3 | 22 001 MiB |
| 32 768 | 131 072 | 131,8 | 25 966 MiB |
| 32 768 | **262 144** | **64,4** | **31 051 MiB** |

Bis 131 072 ist der Durchsatz **identisch** — ein größer konfiguriertes Fenster kostet
Speicher, aber keine Geschwindigkeit. Der Einbruch bei 262 144 ist kein Effekt der
Fenstergröße, sondern des vollen Speichers: mit `q4_0` statt `q8_0` verschwindet er:

| `num_ctx` (`q4_0`, 8k Füllung) | Decode | VRAM |
|---:|---:|---:|
| 131 072 | 131,5 | 23 970 MiB |
| 196 608 | 130,0 | 25 810 MiB |
| 262 144 | 131,9 | 27 567 MiB |

Spanne 1,5 %. **Die Regel lautet: allokiere so groß du willst, solange die Karte nicht
volläuft.**

Praktische Folge für OpenClaw: Da eine Session ein einziger wachsender Kontext ist, läuft
sie am Anfang mit voller Geschwindigkeit und wird erst langsamer, während sie sich füllt.
Die 30 % Einbuße zwischen 131k und 262k treffen euch also nur am Ende einer langen
Session, nicht durchgehend.

---

## 3. Kontext-Kurve bei vollem Füllstand

`q4_0`, Library-Q4_K_M, 512 Decode-Token, 5 Messläufe je Zelle.

| Kontext | Decode | Prefill | VRAM | Grenzkosten je +32k |
|---:|---:|---:|---:|---:|
| 131 072 | **76,5** | 1 509 | 20 904 MiB | — |
| 163 840 | 67,7 | 1 307 | 21 832 MiB | −8,8 tok/s |
| 196 608 | 62,9 | 1 112 | 22 760 MiB | −4,7 tok/s |
| 262 144 | **53,6** | 940 | 24 616 MiB | −4,7 tok/s |

Der **erste** Schritt weg von 128k ist doppelt so teuer wie alle folgenden. Deshalb lohnen
die Zwischenstufen wenig: 160k kostet 11,5 % für 25 % mehr Fenster, 256k kostet 30 % für
100 %. Wer den ersten Schritt zahlt, sollte durchgehen.

Prefill-Zeit für ein volles Fenster: 87 s bei 131k, 125 s bei 164k, 177 s bei 197k,
276 s bei 262k. Das fällt nur bei Cache-Miss an — bei OpenClaw also an Sessionstart und
nach jedem Compact, nicht pro Zug.

---

## 4. KV-Stufen im direkten Vergleich

Bei 131 072, identisches Protokoll:

| KV | Decode | Prefill | VRAM | KV-Anteil |
|---|---:|---:|---:|---:|
| `f16` | 61,7 | 598 | 26 322 MiB | 8 192 MiB |
| `q8_0` | **79,5** | 1 551 | 22 952 MiB | 4 352 MiB |
| `q4_0` | 76,5 | 1 509 | **20 904 MiB** | 2 304 MiB |

`q4_0` ist **nicht schneller**, sondern 3,8 % langsamer als `q8_0` — kontraintuitiv, aber
plausibel: die Dequantisierung je Zugriff kostet mehr, als die eingesparte Bandbreite
bringt. Der Handel lautet 2 GB gegen 4 %.

Unterhalb von 32k spielt der KV-Typ praktisch keine Rolle (alle Stufen 105–117 tok/s).

**Qualität: in allen 19 Zellen makellos.** Needle 5/5 über fünf Tiefen, Multi-Needle 5/5,
Verbatim-Codeblock zeichengenau reproduziert — bei jedem KV-Typ und jeder Kontextlänge bis
260 809 Token. Kein einziger Messpunkt zeigt Degradation. Die Sorge, ein Qwen-Modell könne
bei aggressiver KV-Quantisierung einbrechen, bestätigt sich auf dem llama.cpp-Pfad nicht.

*Einschränkung:* Die Sonden messen Retrieval und Reproduktion, nicht Perplexität.
Schleichende Verschlechterung in freier Generierung wäre damit nicht zwingend sichtbar.

---

## 5. Produktivbudget: der Embedder und die Verdrängung

Gemessen: Embedder `qwen3-embedding:4b-ctx2k` = **4 231 MiB** (Doku sagte ~3 800).
Desktop-Grundlast **1 996 MiB** im Normalzustand, beobachtete Spanne 1 114–4 414 MiB.

Beide Modelle koresident, identischer Prompt, kurze Session:

| Chat-Fenster | Decode | VRAM belegt | frei | Shared |
|---|---:|---:|---:|---:|
| 131 072 | **94,2** tok/s | 28 103 MiB | 4 085 MiB | 657 MiB *(Basis)* |
| 262 144 | 90,0 tok/s | 31 145 MiB | **1 043 MiB** | 1 417 MiB *(Spilling)* |

**Ollamas Scheduler verdrängt den Embedder**, sobald das große Modell frisch geladen wird
— auch bei `keep_alive: -1` und auch dann, wenn Platz wäre. Ein anschließender
Embedding-Aufruf holt ihn wieder dazu. Im Wechselbetrieb kostet das Reload-Latenz, führt
aber nicht zum Kollaps.

Maximal mögliche Kontextlänge, aus dem linearen Fit `17 797 MiB + 29,1 KiB × Kontext`:

| Desktop | frei fürs Chat-Modell | max. Kontext | mit 1 GB Reserve |
|---|---:|---:|---:|
| ruhig (1 996 MiB) | 26 380 MiB | 302 000 | 266 000 |
| belebt (4 414 MiB) | 23 962 MiB | 217 000 | 181 000 |

---

## 6. Ollama allokiert vorab — ein Upgrade ändert das nicht

Sonde mit 6-Token-Prompt, nur `num_ctx` variiert:

| `num_ctx` | 0.32.14 *(euer Stand)* | 0.33.0 |
|---:|---:|---:|
| 8 192 | 18 569 MB | 18 483 MB |
| 32 768 | 19 672 MB | 19 607 MB |
| 131 072 | 23 578 MB | 23 566 MB |

Innerhalb des Rauschens identisch. Die Release Notes bis 0.33.0 nennen keine Änderung an
der Allokation. 0.32.15 bringt Metadaten-Caching mit etwa halbierter TTFT — ein eigener
Upgrade-Grund, aber kein Speichergewinn.

---

## 7. Laufzeiten (für die Planung weiterer Läufe)

| Zelle | Kontexte | Läufe | Dauer |
|---|---|---:|---:|
| Library `q8_0` | 8k/32k/128k | 2 | 24 min |
| Library `q4_0` | 8k/32k/128k | 2 | 16 min |
| Library `f16` | 8k/32k/128k | 2 | 52 min |
| UD-Q6_K `f16` | 8k/32k/128k | 2 | 69 min |
| Library `q8_0` | 32k/128k | 5 | 42 min |
| Library `q4_0` | 131k/164k/197k/262k | 5 | 171 min |
| Allokationstest | 7 Zellen, 8k/32k Füllung | 5 | 14 min |

**Faustformel:** eine Zelle fährt ~12 volle Prefills (1 Warmup + n Messläufe + 5
Needle-Tiefen + 4 Sonden).

```
Zelldauer ≈ 12 × (Kontextlänge / Prefill-tok/s) + ~2 min Modell-Load
```

| Kontext | ein Prefill | ≈ Zelldauer |
|---:|---:|---:|
| 8 192 | 3 s | ~3 min |
| 32 768 | 12 s | ~4 min |
| 131 072 | 83 s | ~19 min |
| 262 144 | 264 s | ~55 min |

Weitere Posten: GGUF-Download 55 GiB ≈ 10 min (in WSL, ~75 MB/s; über den
`//wsl.localhost`-Mount deutlich langsamer). `ollama create` je Modell ≈ 3 min.
Docker-Pull Ollama ≈ 2 min, vLLM-Nightly (29 GB) ≈ 12 min. Container-Neustart ≈ 20 s.

---

## 8. Offen

**O1 · NVFP4 unter vLLM** (~1 h, alles heruntergeladen) — die eine Zahl, die zählt: tok/s
bei 128k gegen die 79,5 hier. Siehe Abschnitt 9.

**O2 · UD-Quant im neuen Protokoll** (~45 min) — der Vorsprung von 5–10 % war über drei
Kontexte konsistent, ist aber nur mit 128 Decode-Token gemessen. Relevant nur, falls ein
Modellwechsel ernsthaft erwogen wird; der Speichervorteil beträgt lediglich 334 MiB.

**O3 · Desktop-Overhead senken** (~15 min) — Browser per *Einstellungen → Grafik* auf die
iGPU legen, NVIDIA-Overlay abschalten. Der Monitor-Umzug auf den iGPU-Ausgang scheidet
aus: **G-Sync funktioniert über Hybrid-Ausgabe nicht** (kein MUX auf Desktop-Boards), und
die Maschine wird auch zum Spielen genutzt.

**O4 · Nativ Ubuntu** (~3 h) — nur noch für den Sondermodus relevant, nicht für den
Alltag: ein Reboot für eine kurze Coding-Session lohnt nicht. Der dokumentierte Vorsprung
von +16 % stammt aus einem 22-Token-Prompt mit **leerem** KV-Cache und ist bei gefülltem
Kontext unbelegt.

**O5 · Perplexität** — die Sonden sind bislang überall perfekt, was den Verdacht nährt,
dass sie zu grob für schleichende Verschlechterung sind. Bräuchte llama.cpp mit CUDA;
ob Upstream die `qwen35`-Architektur kennt, ist ungeprüft.

---

## 9. Recherche: vLLM und NVFP4

**vLLM unterstützt das Modell offiziell** (eigene Recipe, min. 0.17.0). MTP-Draft
funktioniert dort mit Akzeptanzraten von 0,754–0,897 — deutlich über den 0,39–0,86 hier
im Ollama-Log. Dazu **PagedAttention**: der KV-Cache wird in 16-Token-Blöcken aus einem
Pool zugeteilt statt je Slot vorab reserviert. Ein Server bedient damit kurze und lange
Anfragen ohne Reload — was Ollama nicht kann. Und **Sleep Mode** (`POST /sleep`,
`/wake_up`) gibt über 90 % des VRAM frei und weckt große Modelle in 3–6 s.

**Der Haken sind die Gewichte, nicht das Cache-Management:**

| Checkpoint | Gewichte | MTP |
|---|---:|---|
| GGUF Q4_K_M *(aktuell)* | **15,33 GiB** | ja |
| gittensor NVFP4-RTX5090 | 17,48 GiB | nein |
| **RedHatAI INT4** | 18,12 GiB | **ja** |
| QUASAR NVFP4 | 18,36 GiB | nein |
| cyankiwi AWQ-INT4 | 19,57 GiB | nein |
| RadixArk NVFP4 | 20,42 GiB | nein |
| **unsloth NVFP4** *(heruntergeladen)* | 21,81 GiB | ja |

„NVFP4" ist bei unsloth mixed precision — `U8`, `F8_E4M3` und `BF16`, effektiv ~6,4 Bit
statt 4. Dazu kommt vLLMs KV-Untergrenze von FP8 (32 KiB/Token gegen 18 bei `q4_0`).

Mit 8 GB freigehalten für Jarvis und fish-speech:

| | nutzbarer Kontext |
|---|---:|
| Ollama Q4_K_M | **~173 000** |
| RedHatAI INT4 | ~80 000 |
| unsloth NVFP4 | ~25 000 |

Für Koexistenz mit anderen Diensten bleibt Ollama vorn — schlicht wegen der 15,33 GiB.
Für einen dedizierten LLM-Betrieb wäre RedHatAI INT4 mit PagedAttention und MTP der
interessantere Kandidat.

Auf einer einzelnen 5090 ist `--enforce-eager` Pflicht (sonst OOM im CUDA-Graph-Capture),
was ausgerechnet Durchsatz kostet. `--language-model-only` gibt Vision auf und hebt die
KV-Kapazität von 91k auf 136k Token, `--max-num-seqs 8` auf 153k.

Quellen: [vLLM Recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) ·
[Sleep Mode](https://docs.vllm.ai/en/latest/features/sleep_mode/) ·
[unsloth NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)

---

## 10. Messfallen (alle real aufgetreten)

**Prompt-Cache verfälscht den Prefill.** Wiederholt ein Messlauf den Warmup-Prompt, meldet
Ollama die volle `prompt_eval_count`, aber nur die Dauer des ungecachten Rests. Gemessen:
213 920 statt 1 570 tok/s. *Abhilfe:* eindeutiger Prompt-Präfix je Lauf.

**Thinking frisst das Antwortbudget.** qwen3.8 gibt Reasoning im Feld `thinking` aus. Bei
`num_predict 24` bleibt `response` leer — der Needle-Test zählte 0/5, obwohl das Modell
korrekt arbeitete. *Abhilfe:* `think: false` für Retrieval-Sonden.

**Zeitbasierte Log-Fenster addieren Ladevorgänge.** Bei zu weitem `docker logs --since`
summierte der Parser mehrere Model-Loads: 36 120 statt 18 060 MiB. *Abhilfe:* auf den
letzten `llama_model_loader:`-Block zuschneiden.

**`--kv-cache-type` ist nur eine Beschriftung.** Wirksam ist `OLLAMA_KV_CACHE_TYPE` im
Container. Ein per `TaskStop` vermeintlich beendeter Lauf lief als **Waisenprozess** weiter
und konfigurierte den Container unter dem zweiten Lauf um — fünf Zellen wiesen `q4_0` aus,
liefen aber mit `q8_0`. *Abhilfe:* `validate_runs.py` prüft die KV-Puffergröße gegen den
ausgewiesenen Typ, `bench_ollama.py` tut dasselbe inline. **Vor jedem Lauf prüfen, dass
kein zweiter Messprozess läuft** — `run_curve.sh` tut das jetzt selbst.

**Zu kurze Decode-Fenster.** 128 Token bei ~100 tok/s sind 1,3 Sekunden — zu wenig, um die
schwankende MTP-Akzeptanz auszumitteln (Streuung bis 29 %). *Abhilfe:* 512 Token. Der
Wechsel hat die Werte nicht verschoben (79,5 gegen 80,2 und 80,8 bei derselben Zelle), nur
die Streuung reduziert.

**MTP-Durchsatz hängt stark am Prompt.** Repetitiver Fülltext erzeugt hohe Akzeptanz
(131 tok/s), natürliche Prosa niedrigere (94 tok/s) — bei identischer Konfiguration.
Vergleiche nur zwischen Zellen mit demselben Prompt. Derselbe Effekt erklärt, warum euer
`UBUNTU_AB_TEST.md` mit „Zaehle von 1 bis 50" auf 156 tok/s kommt.

**Erste Generierung nach dem Laden läuft ohne Draft.** 48 statt 131 tok/s — kein
Speicherproblem, sondern ein Kaltstart-Artefakt. Steht auch in `UBUNTU_AB_TEST.md`.

**Ollamas `/api/ps` unterschlägt Speicher.** 17 044 statt 22 952 MiB — Compute-Puffer und
CLIP fehlen. Die llama.cpp-Logzeilen sind maßgeblich.

**Gerätweites `nvidia-smi` enthält den Desktop.** Schwankte um bis zu 3 GB. Nur als
Gegenprobe (Differenz zu den Puffersummen ≈ 630 MiB CUDA-Kontext).

**Unfairer Modellvergleich durch unvollständigen Import.** Die ersten UD-Modelle wurden mit
`FROM <gguf>` allein registriert — ohne Vision-Projektor, `RENDERER`/`PARSER` und ohne
`draft_num_predict`, mit dem die Library-Variante spekulatives Decoding fährt. Ergebnis war
ein scheinbarer UD-Rückstand von 40–60 %, der nach korrekter Registrierung in einen
Vorsprung von 5–8 tok/s umschlug. *Abhilfe:* Modelfile der Referenz mit `/api/show`
auslesen und nachbilden.

---

## 11. Verzeichnisse

| Pfad | Inhalt |
|---|---|
| `raw/` | 19 gültige Messläufe, von `validate_runs.py` geprüft |
| `raw_superseded/` | UD-Läufe ohne Draft/Vision — Qualität und KV-Größen gültig, Decode und VRAM-Vergleich nicht |
| `raw_mislabeled/` | Zellen mit falschem KV-Etikett aus dem Parallellauf — unbrauchbar |
| `raw_vllm/` | vLLM-Läufe (noch leer) |
| `budget_*.json`, `allocation_*.json` | Budget- und Allokationsmessungen |
| `*.log` | Laufprotokolle |
