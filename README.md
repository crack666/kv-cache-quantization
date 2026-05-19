# KV-Cache Quantization for Long-Context LLMs

**Thesis**: Long-Context-Effizienz durch KV-Cache-Quantisierung bei Large Language Models  
**Author**: Lennart Behr (Medieninformatik M.Sc., BHT Berlin)  
**Supervisor**: Prof. Dr. Edlich  
**Hardware**: NVIDIA RTX 5090 (32 GB VRAM)

---

## Overview

Profiling code and measurement data for the master thesis on KV-Cache quantization. We systematically evaluate INT8, INT4, and INT2 quantization of the Key-Value Cache across four LLMs with different Grouped Query Attention (GQA) architectures — measuring memory, latency, perplexity, and long-context retrieval (Needle-in-a-Haystack).

### Key Findings (WisPro Phase)

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
├── scripts/
│   ├── profiler_suite.py           # Main CLI: profiling + benchmarks (PPL, Needle)
│   ├── benchmarks/
│   │   ├── needle_haystack.py      # Needle-in-a-Haystack (RULER-style echo-prompt)
│   │   └── perplexity.py           # WikiText-2 / PG-19 sliding-window PPL
│   ├── core/
│   │   ├── model_loader.py         # HF model loading with KV-quant config
│   │   ├── kv_cache.py             # Cache patching, size measurement, timings
│   │   ├── vram_profiler.py        # CUDA peak memory tracking
│   │   └── metrics.py              # Prefill latency, decode throughput
│   ├── aggregate_results.py        # Combine JSON results
│   ├── analyze_delta_ppl.py        # PPL degradation analysis
│   └── generate_*.py               # Table/figure generation
├── results/
│   ├── raw/                        # JSON measurement files
│   │   └── long_context/           # MA phase: 4k–32k context runs
│   ├── figures/                    # Generated plots (PDF)
│   └── tables/                     # LaTeX tables
├── requirements.txt
├── environment.yml
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
# Full profiling + benchmarks for one model
python scripts/profiler_suite.py \
  --model mistralai/Mistral-7B-v0.1 \
  --attn-backend sdpa \
  --kv-quant none int8-hqq int4-hqq int2-hqq int2-hqq-kivi \
  --context-lengths 4096 8192 16384 32768 \
  --benchmarks ppl needle \
  --output-dir results/raw/long_context/ \
  --seed 42

# Minimal: profile one model, FP16 only
python scripts/profiler_suite.py --model gpt2 --context-lengths 128 256
```

**Output:** One JSON per (backend × kv-quant) combination containing:
- Measurements per context length (prefill, decode, VRAM, KV size, power)
- Perplexity (reference + quantized + delta)
- Needle-in-a-Haystack trials (per depth × context length)

### Power Measurement

GPU power sampling via NVML is **enabled by default** on CUDA devices. A background thread samples `nvmlDeviceGetPowerUsage()` at 20 Hz during decode — the overhead is negligible (<0.1% CPU, no GPU impact).

Output fields per context length:
- `avg_power_watts` — mean GPU power draw during decode (Watts)
- `energy_mj_per_token` — energy per generated token (millijoules)

To disable (e.g. on systems without NVML): `--no-measure-power`

### 3. Patch Benchmarks (re-run without re-profiling)

If a benchmark needs to be re-run (e.g. after fixing the needle prompt), patch existing result files without repeating expensive profiling:

```bash
# Re-run needle on all Mistral results
python scripts/profiler_suite.py \
  --model mistralai/Mistral-7B-v0.1 \
  --patch results/raw/long_context/mistral_7b_*.json \
  --benchmarks needle

# Re-run PPL on a specific file, no backup
python scripts/profiler_suite.py \
  --model mistralai/Mistral-7B-v0.1 \
  --patch results/raw/long_context/mistral_7b_v0.1_sdpa_fp16_20260502_152932.json \
  --benchmarks ppl --no-backup
```

### 4. Analyze Results

```bash
python scripts/aggregate_results.py
python scripts/analyze_delta_ppl.py
```

---

## Quantization Backend

We use **HQQ (Half-Quadratic Quantization)** via HuggingFace Transformers:

| Config | Bits | Group Size | Axis | Mode |
|--------|------|-----------|------|------|
| `int8-hqq` | 8 | 64 | 0 | Symmetric |
| `int4-hqq` | 4 | 64 | 0 | Symmetric |
| `int2-hqq` | 2 | 16 | 0 | Symmetric |
| `int2-hqq-kivi` | 2 | 16 | Keys: 0, Values: 1 | Asymmetric (KIVI) |

**Residual Length**: 128 tokens (last 128 KV entries remain in FP16).

---

## Needle-in-a-Haystack Benchmark

Tests whether KV-cache quantization destroys long-range retrieval ability.

- **Format**: Completion/echo-prompt (RULER-style, base-model friendly)
- **Needle**: `"The special magic number for this experiment is 7492."`
- **Prompt**: `"The special magic number for this experiment is"` (model completes)
- **Scoring**: Case-insensitive string match (`"7492" in output`)
- **Reference**: Hsieh et al. (2024), "RULER", COLM 2024

---

## CLI Reference (`profiler_suite.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | HuggingFace model ID or local path |
| `--attn-backend` | `sdpa` | Attention backend(s): `sdpa`, `eager`, `sage` |
| `--kv-quant` | `none` | Quant config(s): `none`, `int8-hqq`, `int4-hqq`, `int2-hqq`, `int2-hqq-kivi` |
| `--context-lengths` | `512 1024 4096` | Context lengths to profile |
| `--benchmarks` | *(none)* | Benchmarks to run: `ppl`, `needle` |
| `--warmup-runs` | `2` | Warmup iterations before measuring |
| `--measure-runs` | `5` | Timed measurement iterations |
| `--decode-tokens` | `128` | Tokens to generate for decode throughput |
| `--no-measure-power` | `false` | Disable GPU power sampling |
| `--residual-length` | `128` | FP16 residual buffer length (KIVI) |
| `--output-dir` | `results/raw/` | Output directory for JSON files |
| `--summary-file` | *(none)* | Compact summary JSON path |
| `--patch` | *(none)* | Patch existing JSON(s) with new benchmarks |
| `--seed` | `42` | Random seed for reproducibility |

---

## Practical Recommendations

- **INT8**: Universally safe (<1% PPL degradation, no retrieval loss)
- **INT4**: Requires model-specific validation
- **INT2**: Only for robust models (Mistral-7B, Yi-1.5-9B)

---

## Citation

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
