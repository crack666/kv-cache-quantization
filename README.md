# KV-Cache Quantization for Long-Context Efficiency

**Masterarbeit**: Systematischer Vergleich von 16/8/4/2-Bit KV-Cache-Quantisierung für Large Language Models  
**Student**: Lennart Behr (Medieninformatik M.Sc., BHT Berlin)  
**Hardware**: NVIDIA RTX 5090 (32 GB VRAM)

---

## 📋 Projektübersicht

Dieses Repository enthält den **Code und die Experimente** für die Masterarbeit zu KV-Cache-Quantisierung. Ziel: Pareto-Analyse (Accuracy vs. VRAM) über verschiedene Quantisierungsmethoden (FP16/INT8/INT4/INT2) auf Long-Context-Benchmarks.

**Kernfragen**:
- Wie viel VRAM spart INT8/INT4/INT2 KV-Cache bei 4k/8k/16k/32k Context?
- Wie stark sinkt die Accuracy (LongBench, Perplexity)?
- Wann lohnt sich welche Bitbreite? (Guidelines für Practitioners)

---

## � Scripts Overview

### `profile_quant_overhead.py` - Complete KV-Cache Profiling

Measures KV-cache size, perplexity, throughput, and quantization overhead across multiple context lengths.

```bash
# Default: Mistral-7B with contexts 128-4096
python scripts/profile_quant_overhead.py

# Custom model
python scripts/profile_quant_overhead.py --model Qwen/Qwen2-7B

# Specific context lengths only (e.g., fill gaps in existing data)
python scripts/profile_quant_overhead.py --model Qwen/Qwen2-7B --context 512 1024 2048

# Custom output path
python scripts/profile_quant_overhead.py --output results/my_profile.json
```

**Output:** JSON file in `results/raw/profile_<model>_<timestamp>.json` with:
- KV-cache sizes (MB) for FP16/INT8/INT4/INT2
- Perplexity (PPL) for quality assessment
- Throughput (tokens/s) and quantization overhead (%)
- Power consumption (watts) and energy per token (mJ/tok)

**Duration:** ~30-60 seconds for 6 contexts × 4 configs = 24 measurements

---

## �🗂️ Repository-Struktur

```
.
├── scripts/               # Python-Skripte für Training, Profiling, Benchmarking
│   ├── profiler.py        # VRAM/Latenz/Throughput-Tracking
│   ├── train_nanogpt.py   # nanoGPT Training (Week 1-2)
│   ├── benchmark_runner.py # lm-eval Wrapper mit Profiling
│   ├── quantize_kv_*.py   # INT8/INT4/INT2 Quantisierung
│   └── plot_*.py          # Visualisierungen (Pareto, VRAM-Scaling, etc.)
├── experiments/           # Wöchentliche Logs (Markdown)
│   ├── week1_2_nanogpt.md
│   ├── week3_4_7b_baseline.md
│   └── ...
├── results/               # Experiment-Daten & Plots
│   ├── raw/               # JSON-Files pro Run (gitignored)
│   ├── figures/           # PDF-Plots (Paper-Quality)
│   ├── tables/            # LaTeX-Tabellen
│   └── master_results.csv # Aggregierte Daten
├── configs/               # JSON-Configs pro Experiment
│   ├── nanogpt_fp16.json
│   ├── mistral7b_int8kv.json
│   └── ...
├── docs/                  # Technische Dokumentation
│   ├── profiling_guide.md # Wie VRAM/Latenz gemessen wird
│   └── quantization_howto.md # INT8/INT4/INT2 Implementierung
├── requirements.txt       # Python-Dependencies
├── environment.yml        # Conda-Environment (reproduzierbar)
├── .gitignore             # Ignoriert große Dateien (Checkpoints, raw results)
└── README.md              # Diese Datei
```

---

## 🚀 Setup

### **1. Environment erstellen**
```bash
# Conda (empfohlen)
conda create -n kv-quant python=3.10
conda activate kv-quant

# Oder venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### **2. Dependencies installieren**
```bash
pip install -r requirements.txt

# Oder mit Conda
conda env create -f environment.yml
conda activate kv-quant
```

### **3. CUDA/PyTorch testen**
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
# Erwartete Ausgabe: CUDA: True, GPU: NVIDIA GeForce RTX 5090
```

---

## 🔬 Quick Start

### **Week 1-2: nanoGPT Baseline**
```bash
# Training (1-2 Stunden auf RTX 5090)
python scripts/train_nanogpt.py --config configs/nanogpt_fp16.json

# Evaluation: Perplexity + VRAM-Profiling
python scripts/eval_perplexity.py --checkpoint results/nanogpt_fp16.pth

# INT8 KV-Cache-Vergleich
python scripts/train_nanogpt.py --config configs/nanogpt_int8kv.json
```

### **Week 3-4: 7B-Modell (Mistral/Llama)**
```bash
# Baseline: FP16 KV
python scripts/benchmark_runner.py --config configs/mistral7b_fp16kv.json

# INT8 KV
python scripts/benchmark_runner.py --config configs/mistral7b_int8kv.json

# Pareto-Plot generieren
python scripts/plot_pareto.py --results results/master_results.csv
```

---

## 📊 Experiment-Tracking

**JSON-Output pro Run** (automatisch in `results/raw/`):
```json
{
  "experiment_id": "mistral7b_int8kv_16k_2025-10-25T14-30",
  "config": {
    "model": "Mistral-7B-v0.1",
    "kv_bits": 8,
    "context_length": 16384
  },
  "metrics": {
    "perplexity": 12.1,
    "vram_peak_gb": 18.5,
    "latency_p50_ms": 145,
    "tokens_per_sec": 85
  }
}
```

**CSV-Aggregation** (`results/master_results.csv`):
```csv
experiment_id,model,kv_bits,context_len,accuracy,vram_gb,latency_ms
mistral7b_fp16,Mistral-7B,16,16384,0.70,24.3,180
mistral7b_int8,Mistral-7B,8,16384,0.68,18.5,145
```

---

## 📈 Visualisierung

**Paper-Quality Plots** (generiert in `results/figures/`):
- `pareto_accuracy_vram.pdf` – Pareto-Front (16/8/4/2-Bit)
- `vram_scaling.pdf` – VRAM vs. Context-Length
- `layer_sensitivity.pdf` – Heatmap (Layer × Quantization)
- `throughput_latency.pdf` – Scatter (Bitwidth)

**Generierung**:
```bash
python scripts/plot_pareto_final.py
python scripts/plot_vram_scaling.py
python scripts/plot_layer_sensitivity.py
```

---

## 🧪 Reproduzierbarkeit

### **Seeds fixieren**
Alle Skripte nutzen feste Seeds:
```python
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
```

### **Exakte Versionen**
`requirements.txt` pinned Versions:
```
torch==2.1.0+cu121
transformers==4.35.0
pynvml==11.5.0
...
```

### **Hardware-Info loggen**
Jedes Experiment speichert:
- GPU-Modell (via `pynvml`)
- CUDA/cuDNN-Version
- Driver-Version

---

## 📚 Dokumentation

- **Experiments**: `experiments/week*.md` – Wöchentliche Logs mit Plots & Erkenntnissen
- **Profiling-Guide**: `docs/profiling_guide.md` – VRAM/Latenz/Throughput messen
- **Quantization-HowTo**: `docs/quantization_howto.md` – INT8/INT4/INT2 implementieren

---

## 🔗 Related Repositories

- **Dokumentation/Recherche** (privat): Literatur, Glossar, Exposé im parallelen Repo
- **KIVI** (Referenz): https://github.com/jy-yuan/KIVI
- **lm-evaluation-harness**: https://github.com/EleutherAI/lm-evaluation-harness

---

## 📝 Zitation

Falls du diesen Code nutzt, bitte referenziere:

```bibtex
@mastersthesis{behr2025kvcache,
  title={Long-Context-Effizienz durch KV-Cache-Quantisierung},
  author={Behr, Lennart},
  year={2025},
  school={Berliner Hochschule für Technik (BHT)}
}
```

---

## 📄 Lizenz

MIT License (oder nach Absprache mit Betreuer)

---

**Status**: Week 0 abgeschlossen (Exposé), Week 1-2 aktiv (nanoGPT Setup)  
**Letzte Aktualisierung**: 2025-10-24
