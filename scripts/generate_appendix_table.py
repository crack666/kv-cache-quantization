#!/usr/bin/env python3
"""
Generate LaTeX appendix tables from long-context profiling JSON results.

Produces two tables:
1. profiling_table_all.tex  -- Full per-context profiling longtable (appendix)
2. model_metrics_table.tex  -- Static model overview (architecture + kurtosis)

Reads single-run JSONs from results/raw/long_context/ and kurtosis data
from results/raw/kv_distributions_v2/.

Usage:
    python generate_appendix_table.py
    python generate_appendix_table.py --input-dir ../results/raw/long_context
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

MODEL_ORDER = [
    "google/gemma-4-E4B",
    "mistralai/Mistral-7B-v0.1",
    "01-ai/Yi-1.5-9B",
    "Qwen/Qwen2-7B",
    "Qwen/Qwen3-8B",
]

MODEL_SHORT = {
    "google/gemma-4-E4B": "Gemma-4-E4B",
    "mistralai/Mistral-7B-v0.1": "Mistral-7B",
    "01-ai/Yi-1.5-9B": "Yi-1.5-9B",
    "Qwen/Qwen2-7B": "Qwen2-7B",
    "Qwen/Qwen3-8B": "Qwen3-8B",
}

CONFIG_ORDER = ["FP16", "INT8 (HQQ)", "INT4 (HQQ)", "INT2 (HQQ)", "INT2 (KIVI)"]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def config_label(kv_quant: dict) -> str:
    """Convert kv_quant dict to display label."""
    if not kv_quant.get("enabled", False):
        return "FP16"
    nbits = kv_quant.get("nbits")
    asymmetric = kv_quant.get("asymmetric", False)
    if asymmetric and nbits == 2:
        return "INT2 (KIVI)"
    return f"INT{nbits} (HQQ)"


def format_ppl(val) -> str:
    """Format perplexity value for LaTeX (scientific notation for extreme)."""
    if val is None:
        return "--"
    if val >= 10000:
        exp = len(str(int(val))) - 1
        mantissa = val / (10 ** exp)
        return f"${mantissa:.1f} \\times 10^{{{exp}}}$"
    return f"{val:.2f}"


def format_delta_ppl(val) -> str:
    """Format ΔPPL for LaTeX (scientific notation for ≥1000)."""
    if val is None:
        return "--"
    if abs(val) < 0.005:
        return "0.0"
    if abs(val) >= 1000:
        exp = len(str(int(abs(val)))) - 1
        mantissa = val / (10 ** exp)
        sign = "+" if val > 0 else ""
        return f"${sign}{mantissa:.1f} \\times 10^{{{exp}}}$"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.2f}"


def apply_bht_fixes(latex: str) -> str:
    """Apply BHT template compatibility fixes to longtable output."""
    # Remove midrule before endfirsthead/endhead (causes 'Misplaced \\cr')
    latex = latex.replace('\\midrule\n\\endfirsthead\n', '\\endfirsthead\n')
    latex = latex.replace('\\midrule\n\\endhead\n', '\\endhead\n')
    return latex


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_single_run_jsons(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all single-run JSON files (exclude summary files)."""
    results = []
    for json_file in sorted(results_dir.glob("*.json")):
        if "summary" in json_file.name:
            continue
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                results.append(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠ Warning: Could not load {json_file}: {e}")
    return results


def load_kurtosis_jsons(kurtosis_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load kurtosis distribution JSONs, keyed by model name."""
    result = {}
    if not kurtosis_dir.exists():
        print(f"⚠ Kurtosis dir not found: {kurtosis_dir}")
        return result
    for json_file in sorted(kurtosis_dir.glob("kv_dist_*.json")):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                model = data.get("model", "unknown")
                result[model] = data.get("summary", {})
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠ Warning: Could not load {json_file}: {e}")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Table 1: Full profiling longtable
# ──────────────────────────────────────────────────────────────────────────────

def extract_rows(data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract flat rows from all JSON files."""
    rows = []
    for data in data_list:
        model = data.get("model", "unknown")
        model_short = MODEL_SHORT.get(model, model.split("/")[-1])
        config = config_label(data.get("kv_quant", {}))

        benchmarks = data.get("benchmarks", {})
        ppl_baseline = benchmarks.get("perplexity", {}).get("value")
        ppl_quant = benchmarks.get("perplexity_quantized", {}).get("value")

        for m in data.get("measurements", []):
            rows.append({
                "model": model,
                "model_short": model_short,
                "config": config,
                "context_len": m["context_len"],
                "prefill_ms": m.get("prefill_ms", 0),
                "decode_tok_s": m.get("decode_tokens_per_sec", 0),
                "kv_cache_mb": m.get("kv_cache_mb", 0),
                "overhead_pct": m.get("overhead_pct", 0),
                "ppl_baseline": ppl_baseline,
                "ppl_quant": ppl_quant,
            })
    return rows


def compute_delta_ppl(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add ppl and delta_ppl fields."""
    for row in rows:
        if row["config"] == "FP16":
            row["ppl"] = row["ppl_baseline"]
            row["delta_ppl"] = 0.0
        else:
            row["ppl"] = row["ppl_quant"]
            if row["ppl_baseline"] and row["ppl_quant"]:
                row["delta_ppl"] = row["ppl_quant"] - row["ppl_baseline"]
            else:
                row["delta_ppl"] = None
    return rows


def sort_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort by model order, then context length, then config order."""
    def sort_key(row):
        model_idx = MODEL_ORDER.index(row["model"]) if row["model"] in MODEL_ORDER else 99
        config_idx = CONFIG_ORDER.index(row["config"]) if row["config"] in CONFIG_ORDER else 99
        return (model_idx, row["context_len"], config_idx)
    return sorted(rows, key=sort_key)


def generate_profiling_table(rows: List[Dict[str, Any]]) -> str:
    """Generate LaTeX longtable with BHT-compatible formatting."""
    lines = []

    # Use |p{}| column format matching old style
    col_fmt = r"|p{2.0cm}|p{2.0cm}|r|r|r|r|r|>{\raggedleft\arraybackslash}p{1.3cm}|"
    lines.append(f"\\begin{{longtable}}{{{col_fmt}}}")
    lines.append(r"\caption{Vollständige KV-Cache Profiling-Ergebnisse (alle Kontextlängen)}")
    lines.append(r"\label{tab:profiling_all} \\")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Modell} & \textbf{Config} & \textbf{Ctx} & "
        r"\textbf{Prefill (ms)} & \textbf{Decode (tok/s)} & "
        r"\textbf{KV-Cache (MB)} & \textbf{PPL} & "
        r"\textbf{$\Delta$PPL} \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")

    # Continuation header (no extra caption on subsequent pages)
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Modell} & \textbf{Config} & \textbf{Ctx} & "
        r"\textbf{Prefill (ms)} & \textbf{Decode (tok/s)} & "
        r"\textbf{KV-Cache (MB)} & \textbf{PPL} & "
        r"\textbf{$\Delta$PPL} \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\endhead")

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{8}{r}{\textit{Fortsetzung nächste Seite}} \\")
    lines.append(r"\endfoot")

    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    # Data rows grouped by model
    prev_model = None
    for row in rows:
        if prev_model is not None and row["model_short"] != prev_model:
            lines.append(r"\midrule")
        prev_model = row["model_short"]

        ctx_str = f"{row['context_len']//1024}k" if row["context_len"] >= 1024 else str(row["context_len"])

        line = (
            f"{row['model_short']} & "
            f"{row['config']} & "
            f"{ctx_str} & "
            f"{row['prefill_ms']:.0f} & "
            f"{row['decode_tok_s']:.1f} & "
            f"{row['kv_cache_mb']:.1f} & "
            f"{format_ppl(row.get('ppl'))} & "
            f"{format_delta_ppl(row.get('delta_ppl'))} \\\\"
        )
        lines.append(line)

    lines.append(r"\end{longtable}")

    latex = "\n".join(lines)
    latex = apply_bht_fixes(latex)
    return latex


# ──────────────────────────────────────────────────────────────────────────────
# Table 2: Model metrics overview
# ──────────────────────────────────────────────────────────────────────────────

def extract_model_metrics(data_list: List[Dict[str, Any]],
                          kurtosis_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract one row per model with architecture info + kurtosis."""
    seen = {}
    for data in data_list:
        model = data.get("model", "unknown")
        if model in seen:
            continue
        mc = data.get("model_config", {})
        kurt = kurtosis_data.get(model, {})
        seen[model] = {
            "model": model,
            "model_short": MODEL_SHORT.get(model, model.split("/")[-1]),
            "num_params_b": mc.get("num_params_b", 0),
            "num_layers": mc.get("num_layers", 0),
            "num_kv_heads": mc.get("num_kv_heads", 0),
            "gqa_ratio": mc.get("gqa_ratio", "--"),
            "head_dim": mc.get("head_dim", 0),
            "key_kurt_mean": kurt.get("key_kurtosis_mean"),
            "key_kurt_max": kurt.get("key_kurtosis_max"),
            "val_kurt_mean": kurt.get("value_kurtosis_mean"),
            "heavy_tail_layers": kurt.get("layers_with_heavy_tails"),
        }

    # Sort by MODEL_ORDER
    rows = []
    for model in MODEL_ORDER:
        if model in seen:
            rows.append(seen[model])
    return rows


def generate_model_metrics_table(rows: List[Dict[str, Any]]) -> str:
    """Generate a compact LaTeX table with per-model architecture and kurtosis."""
    lines = []

    col_fmt = r"|p{2.2cm}|r|r|r|r|r|r|r|r|"
    lines.append(f"\\begin{{table}}[htbp]")
    lines.append(f"\\centering")
    lines.append(r"\caption{Modellarchitektur und Kurtosis-Kennzahlen}")
    lines.append(r"\label{tab:model_metrics}")
    lines.append(f"\\begin{{tabular}}{{{col_fmt}}}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Modell} & \textbf{Param. (B)} & \textbf{Layers} & "
        r"\textbf{KV-Heads} & \textbf{GQA} & \textbf{$d_h$} & "
        r"\textbf{$\bar{\kappa}_K$} & \textbf{$\kappa_{K,\max}$} & "
        r"\textbf{$\bar{\kappa}_V$} \\"
    )
    lines.append(r"\midrule")

    for row in rows:
        kurt_k = f"{row['key_kurt_mean']:.2f}" if row['key_kurt_mean'] is not None else "--"
        kurt_k_max = f"{row['key_kurt_max']:.2f}" if row['key_kurt_max'] is not None else "--"
        kurt_v = f"{row['val_kurt_mean']:.2f}" if row['val_kurt_mean'] is not None else "--"

        line = (
            f"{row['model_short']} & "
            f"{row['num_params_b']:.2f} & "
            f"{row['num_layers']} & "
            f"{row['num_kv_heads']} & "
            f"{row['gqa_ratio']} & "
            f"{row['head_dim']} & "
            f"{kurt_k} & "
            f"{kurt_k_max} & "
            f"{kurt_v} \\\\"
        )
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX appendix tables from profiling + kurtosis JSONs"
    )
    parser.add_argument(
        "--input-dir", type=Path,
        default=Path(__file__).parent.parent / "results" / "raw" / "long_context",
        help="Directory containing single-run JSON files",
    )
    parser.add_argument(
        "--kurtosis-dir", type=Path,
        default=Path(__file__).parent.parent / "results" / "raw" / "kv_distributions_v2",
        help="Directory containing kurtosis distribution JSONs",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent.parent / "results" / "tables",
        help="Output directory for LaTeX table files",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Reading profiling JSONs from: {args.input_dir}")
    data_list = load_single_run_jsons(args.input_dir)
    print(f"Loaded {len(data_list)} JSON files")

    print(f"Reading kurtosis JSONs from: {args.kurtosis_dir}")
    kurtosis_data = load_kurtosis_jsons(args.kurtosis_dir)
    print(f"Loaded kurtosis data for {len(kurtosis_data)} models")

    # Table 1: Full profiling table
    rows = extract_rows(data_list)
    print(f"Extracted {len(rows)} measurement rows")
    rows = compute_delta_ppl(rows)
    rows = sort_rows(rows)

    profiling_tex = generate_profiling_table(rows)
    profiling_path = args.output_dir / "profiling_table_all.tex"
    with open(profiling_path, "w", encoding="utf-8") as f:
        f.write(profiling_tex)
    print(f"✅ Profiling table: {profiling_path}")
    print(f"   {len(rows)} rows, {len(set(r['model'] for r in rows))} models")

    # Table 2: Model metrics overview
    metrics_rows = extract_model_metrics(data_list, kurtosis_data)
    metrics_tex = generate_model_metrics_table(metrics_rows)
    metrics_path = args.output_dir / "model_metrics_table.tex"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(metrics_tex)
    print(f"✅ Model metrics table: {metrics_path}")
    print(f"   {len(metrics_rows)} models")


if __name__ == "__main__":
    main()
