# KV-Cache Quantization for Long-Context LLMs

**Thesis**: Long-Context-Effizienz durch KV-Cache-Quantisierung bei Large Language Models  
**Author**: Lennart Behr (Medieninformatik M.Sc., BHT Berlin)  
**Supervisor**: Prof. Dr. Edlich  
**Hardware**: NVIDIA RTX 5090 (32 GB VRAM)

---

## Overview

This repository contains the **profiling code and measurement data** for the master thesis on KV-Cache quantization. We systematically evaluate INT8, INT4, and INT2 quantization of the Key-Value Cache across four LLMs with different Grouped Query Attention (GQA) architectures.

### Key Finding

> The GQA ratio is **not** a reliable predictor of quantization tolerance. Yi-1.5-9B (8:1) tolerates INT2 with only +15% PPL degradation, while Qwen2-7B (7:1) fails catastrophically at INT4.

---

## Models Tested

| Model | Parameters | GQA Ratio | INT4 Tolerance | INT2 Tolerance |
|-------|------------|-----------|----------------|----------------|
| Mistral-7B-v0.1 | 7.2B | 4:1 | ✅ Excellent (-1.0% PPL) | ✅ Good (+2.2% PPL) |
| Qwen3-8B | 8.2B | 4:1 | ✅ Good (+7.0% PPL) | ⚠️ Degraded (+96% PPL) |
| Qwen2-7B | 7.6B | 7:1 | ❌ Fails | ❌ Fails |
| Yi-1.5-9B | 8.8B | 8:1 | ✅ Excellent (-0.3% PPL) | ✅ Usable (+15% PPL) |

---

## Repository Structure

```
.
├── scripts/                    # Profiling and analysis scripts
│   ├── profile_quant_overhead.py   # Main profiler (KV-cache, PPL, throughput)
│   ├── aggregate_results.py        # Combine JSON results
│   ├── analyze_delta_ppl.py        # PPL degradation analysis
│   └── generate_*.py               # Table/figure generation
├── results/
│   ├── raw/                    # JSON measurement files (per model)
│   ├── figures/                # Generated plots (PDF)
│   └── tables/                 # LaTeX tables
├── requirements.txt            # Python dependencies
├── environment.yml             # Conda environment
└── README.md
```

---

## Quick Start

### 1. Setup Environment

```bash
conda env create -f environment.yml
conda activate kv-quant
# or
pip install -r requirements.txt
```

### 2. Run Profiling

```bash
# Profile a model across context lengths 128-4096
python scripts/profile_quant_overhead.py --model mistralai/Mistral-7B-v0.1

# Profile specific context lengths
python scripts/profile_quant_overhead.py --model Qwen/Qwen2-7B --context 512 1024 2048
```

**Output:** JSON file in `results/raw/profile_<model>_<timestamp>.json` containing:
- KV-cache sizes (MB) for FP16/INT8/INT4/INT2
- Perplexity for quality assessment
- Throughput (tokens/s) and quantization overhead (%)

### 3. Analyze Results

```bash
# Aggregate all JSON files into summary
python scripts/aggregate_results.py

# Generate PPL degradation analysis
python scripts/analyze_delta_ppl.py
```

---

## Quantization Backend

We use **HQQ (Half-Quadratic Quantization)** via HuggingFace Transformers with:

- **INT8**: Group Size 64, Axis 0
- **INT4**: Group Size 64, Axis 0  
- **INT2**: Group Size 16, Axis 0
- **Residual Length**: 128 tokens (last 128 tokens remain in FP16)

---

## Results Summary

### Memory Reduction

| Bitwidth | KV-Cache Size | Reduction |
|----------|---------------|-----------|
| FP16 | 100% (baseline) | — |
| INT8 | 50% | 2× compression |
| INT4 | 25% | 4× compression |
| INT2 | 12.5% | 8× compression |

### Practical Recommendations

- **INT8**: Universally safe (<1% PPL degradation)
- **INT4**: Requires model-specific validation
- **INT2**: Only for robust models (Mistral-7B, Yi-1.5-9B)

---

## Citation

If you use this code or data, please cite the thesis:

```bibtex
@mastersthesis{behr2026kvcache,
  author = {Behr, Lennart},
  title = {Long-Context-Effizienz durch KV-Cache-Quantisierung bei Large Language Models},
  school = {Berliner Hochschule für Technik},
  year = {2026},
  type = {Master's Thesis}
}
```

---

## License

MIT License. See individual model licenses for usage restrictions.
