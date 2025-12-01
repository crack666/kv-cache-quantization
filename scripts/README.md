# KV-Cache Quantization Scripts - Usage Guide

## Übersicht

Dieses Verzeichnis enthält die Experiment-Skripte für die KV-Cache Quantisierung.

| Skript | Zweck |
|--------|-------|
| `quantize_kvcache_hf.py` | Hauptskript für Quantisierungsexperimente |
| `aggregate_results.py` | Aggregiert und visualisiert Ergebnisse |
| `run_experiments.py` | Orchestrator für Matrix-Experimente |
| `profiler.py` | Basis-Profiler (deprecated, nutze `quantize_kvcache_hf.py`) |

---

## quantize_kvcache_hf.py - Hauptskript

### Grundsyntax

```bash
python quantize_kvcache_hf.py \
    --model <MODEL_NAME> \
    --context-lengths <LEN1> <LEN2> ... \
    [--quantize | --no-quantize] \
    [--nbits {2,4,8}] \
    [--backend {quanto,hqq}] \
    [--seed <SEED>] \
    [--output-dir <DIR>]
```

### Parameter

| Parameter | Pflicht | Default | Beschreibung |
|-----------|---------|---------|--------------|
| `--model` | ✅ | - | HuggingFace Model-ID (z.B. `gpt2`, `mistralai/Mistral-7B-v0.1`) |
| `--context-lengths` | ✅ | - | Liste von Kontextlängen zum Testen |
| `--quantize` | ❌ | False | Aktiviert Quantisierung |
| `--no-quantize` | ❌ | True | FP16 Baseline (kein Quantisierung) |
| `--nbits` | ❌ | 4 | Bit-Präzision: 2, 4 oder 8 |
| `--backend` | ❌ | quanto | Backend: `quanto` oder `hqq` |
| `--seed` | ❌ | 42 | Random Seed für Reproduzierbarkeit |
| `--output-dir` | ❌ | `../results/raw` | Ausgabeverzeichnis für JSON-Dateien |

---

## Beispiel-Workflows

### 1. FP16 Baseline (keine Quantisierung)

```bash
# GPT-2 FP16 Baseline
python quantize_kvcache_hf.py \
    --model gpt2 \
    --context-lengths 128 256 512 \
    --no-quantize

# Mistral-7B FP16 Baseline mit langen Kontexten
python quantize_kvcache_hf.py \
    --model "mistralai/Mistral-7B-v0.1" \
    --context-lengths 128 256 512 1024 2048 4096 8192 16384 \
    --no-quantize
```

### 2. INT8 Quantisierung (HQQ Backend)

```bash
# GPT-2 INT8
python quantize_kvcache_hf.py \
    --model gpt2 \
    --context-lengths 128 256 512 \
    --quantize --nbits 8 --backend hqq

# Mistral-7B INT8
python quantize_kvcache_hf.py \
    --model "mistralai/Mistral-7B-v0.1" \
    --context-lengths 128 256 512 1024 2048 4096 \
    --quantize --nbits 8 --backend hqq
```

### 3. INT4 Quantisierung

```bash
# INT4 mit HQQ (empfohlen - echte 4x Kompression)
python quantize_kvcache_hf.py \
    --model "mistralai/Mistral-7B-v0.1" \
    --context-lengths 128 256 512 1024 \
    --quantize --nbits 4 --backend hqq

# INT4 mit quanto (nur 2x Kompression wegen FP16-Scales)
python quantize_kvcache_hf.py \
    --model "mistralai/Mistral-7B-v0.1" \
    --context-lengths 128 256 512 1024 \
    --quantize --nbits 4 --backend quanto
```

### 4. INT2 Quantisierung (aggressiv)

```bash
# INT2 mit HQQ (8x Kompression)
python quantize_kvcache_hf.py \
    --model "mistralai/Mistral-7B-v0.1" \
    --context-lengths 128 256 512 1024 \
    --quantize --nbits 2 --backend hqq
```

---

## Dateinamen-Konvention

Jeder Run erzeugt eine neue JSON-Datei mit folgendem Schema:

```
<MODEL>_<PRECISION>_<TIMESTAMP>.json
```

**Beispiele:**
```
gpt2_fp16_20251129_051528.json           # GPT-2 FP16 Baseline
gpt2_int4_20251129_051542.json           # GPT-2 INT4 (Backend im JSON)
mistralai_Mistral-7B-v0.1_int2_20251129_055711.json  # Mistral INT2
```

**Wichtig:** Der Dateiname enthält NICHT das Backend (hqq/quanto). Dieses ist im JSON unter `config.backend` gespeichert.

---

## JSON-Ausgabeformat (Schema v2.0)

```json
{
  "experiment_id": "mistralai_Mistral-7B-v0.1_int2_20251129_055711",
  "schema_version": "2.0",
  
  "config": {
    "model": "mistralai/Mistral-7B-v0.1",
    "kv_precision": "int2",         // fp16, int8, int4, int2
    "nbits": 2,                     // 16, 8, 4, 2
    "backend": "hqq",               // none, hqq, quanto
    "method": "baseline",
    "device": "cuda",
    "seed": 42,
    "context_lengths": [128, 256, 512, 1024]
  },
  
  "environment": {
    "python_version": "3.10.13",
    "pytorch_version": "2.9.1+cu128",
    "transformers_version": "4.57.1",
    "cuda_version": "12.8",
    "gpu_name": "NVIDIA GeForce RTX 5090",
    "gpu_memory_gb": 34.2
  },
  
  "measurements": [
    {
      "context_length": 128,
      "kv_cache": {
        "total_bytes": 2097152,      // Absolut in Bytes
        "total_gb": 0.002097152,
        "bytes_per_token": 16384.0   // Key-Metrik für Kompression!
      },
      "vram": {
        "before_gb": 21.05,
        "after_gb": 21.19,
        "delta_gb": 0.14
      },
      "latency_ms": 644.88,
      "perplexity": 1.112,           // Key-Metrik für Qualität!
      "timestamp": "2025-11-29T14:34:21.201849"
    }
    // ... weitere Context-Lengths
  ],
  
  "summary": {
    "avg_bytes_per_token": 16384.0,
    "avg_perplexity": 1.124,
    "compression_ratio": 8.0         // Relativ zu FP16 (131072/16384)
  }
}
```

---

## Interpretation der Ergebnisse

### Key-Metriken

| Metrik | Bedeutung | Gut wenn... |
|--------|-----------|-------------|
| `bytes_per_token` | KV-Cache Größe pro Token | Niedriger = bessere Kompression |
| `perplexity` | Modell-Unsicherheit | Nahe am FP16-Baseline |
| `compression_ratio` | FP16 / quantized | Höher = besser |
| `latency_ms` | Inferenz-Latenz | Konstant (Overhead akzeptabel) |

### Kompressionsberechnung

```
Kompression = FP16_bytes_per_token / Quantized_bytes_per_token
```

**Beispiel Mistral-7B:**
- FP16: 131,072 bytes/token
- INT4-hqq: 32,768 bytes/token → 4× Kompression
- INT2-hqq: 16,384 bytes/token → 8× Kompression

### PPL-Degradation berechnen

```
Δ PPL = (PPL_quantized - PPL_fp16) / PPL_fp16 × 100%
```

**Beispiel:**
- FP16 PPL: 1.112
- INT2 PPL: 1.116
- Δ PPL = (1.116 - 1.112) / 1.112 × 100% = +0.4%

---

## aggregate_results.py - Ergebnis-Aggregation

Aggregiert alle JSON-Dateien und erstellt übersichtliche Tabellen:

```bash
# Markdown-Tabelle (Standard)
python aggregate_results.py --table

# Nur neueste pro Konfiguration
python aggregate_results.py --table --latest

# LaTeX-Tabelle für Thesis
python aggregate_results.py --latex

# Alle Formate + Summary
python aggregate_results.py --all --latest

# In Datei speichern
python aggregate_results.py --latex --output ../results/tables/results.tex
```

---

## Reproduzierbarkeit

### Vollständige Experiment-Matrix nachstellen

```bash
# 1. Umgebung prüfen
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# 2. Alle 4 Modelle × 5 Konfigurationen testen
MODELS=("gpt2" "Qwen/Qwen2-0.5B" "Qwen/Qwen2-7B" "mistralai/Mistral-7B-v0.1")
CONTEXT="128 256 512"

for model in "${MODELS[@]}"; do
    echo "=== Testing $model ==="
    
    # FP16 Baseline
    python quantize_kvcache_hf.py --model "$model" --context-lengths $CONTEXT --no-quantize
    
    # INT8 HQQ
    python quantize_kvcache_hf.py --model "$model" --context-lengths $CONTEXT --quantize --nbits 8 --backend hqq
    
    # INT4 HQQ
    python quantize_kvcache_hf.py --model "$model" --context-lengths $CONTEXT --quantize --nbits 4 --backend hqq
    
    # INT4 quanto
    python quantize_kvcache_hf.py --model "$model" --context-lengths $CONTEXT --quantize --nbits 4 --backend quanto
    
    # INT2 HQQ
    python quantize_kvcache_hf.py --model "$model" --context-lengths $CONTEXT --quantize --nbits 2 --backend hqq
    
    # INT2 quanto
    python quantize_kvcache_hf.py --model "$model" --context-lengths $CONTEXT --quantize --nbits 2 --backend quanto
done

# 3. Ergebnisse aggregieren
python aggregate_results.py --all --latest
```

### Native Ubuntu vs WSL

Für den Vergleich Windows-WSL vs. Native Ubuntu:

1. **Gleiche Umgebung sicherstellen:**
   ```bash
   conda env export > environment.yml
   pip freeze > requirements.txt
   ```

2. **Auf Ubuntu reproduzieren:**
   ```bash
   conda env create -f environment.yml
   # oder
   pip install -r requirements.txt
   ```

3. **Ergebnisse mit unterschiedlichem Prefix speichern:**
   ```bash
   # WSL
   python quantize_kvcache_hf.py ... --output-dir ../results/raw/wsl/
   
   # Native Ubuntu
   python quantize_kvcache_hf.py ... --output-dir ../results/raw/native/
   ```

---

## Bekannte Limitationen

### Modell-spezifische Context-Limits

| Modell | Max Context | Grund |
|--------|-------------|-------|
| GPT-2 | 1024 | Position Embedding Limit |
| Qwen2 | 131,072 | VRAM-limitiert (~32K praktisch) |
| Mistral-7B | 32,768 | VRAM-limitiert (~16K auf 32GB) |

### Backend-Unterschiede

| Backend | INT2 | INT4 | INT8 | Kompression |
|---------|------|------|------|-------------|
| **hqq** | ✅ | ✅ | ✅ | Theoretisch optimal (8×/4×/2×) |
| **quanto** | ✅ | ✅ | ❌ | Nur 2× wegen FP16-Scales |

### Qwen-Inkompatibilität

⚠️ **Qwen-Modelle sind NICHT kompatibel mit INT4/INT2 Quantisierung!**

| Modell | INT8 | INT4 | INT2 |
|--------|------|------|------|
| Qwen2-0.5B | +0.3% ✅ | +449% ❌ | +44% ⚠️ |
| Qwen2-7B | +0.0% ✅ | +2.8M% ❌ | +134M% ❌ |

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Kontextlänge reduzieren
python quantize_kvcache_hf.py --model "..." --context-lengths 128 256 512 ...

# GPU-Cache leeren vor dem Run
python -c "import torch; torch.cuda.empty_cache()"
```

### "QuantizedCache not found"

```bash
# Transformers Version prüfen (muss >= 4.43 sein)
pip install transformers>=4.43
```

### Quantisierung wird nicht angewendet

**Problem:** Datei zeigt `fp16` obwohl `--nbits 4` angegeben wurde.

**Lösung:** `--quantize` Flag vergessen!
```bash
# FALSCH (Quantisierung wird ignoriert):
python quantize_kvcache_hf.py --model gpt2 --nbits 4 --context-lengths 128 256

# RICHTIG:
python quantize_kvcache_hf.py --model gpt2 --quantize --nbits 4 --context-lengths 128 256
```

---

## Autor

WisPro Projekt - KV-Cache Quantisierung für Long-Context LLMs  
November 2025
