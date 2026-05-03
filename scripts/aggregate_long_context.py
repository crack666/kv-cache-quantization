#!/usr/bin/env python3
"""Cross-model aggregation of Phase B long-context profiling results.

Reads all individual combo JSONs from results/raw/long_context/ and produces:
  1. A unified CSV (one row per model × kv_quant × context_length)
  2. A compact LaTeX-ready table (Δ-PPL comparison @ max context)
  3. Console summary

Usage:
    python aggregate_long_context.py [--output-dir ../results/tables/]
"""
import argparse
import csv
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "raw" / "long_context"


def load_all_results(folder: Path) -> list[dict]:
    """Load all individual combo JSONs (skip summaries)."""
    results = []
    for p in sorted(folder.glob("*.json")):
        if "summary" in p.name:
            continue
        try:
            results.append(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARN: skipping {p.name} — {e}")
    return results


def kv_quant_label(kv: dict) -> str:
    """Human-readable KV-quant label."""
    if not kv["enabled"]:
        return "FP16"
    label = f"INT{kv['nbits']}"
    if kv.get("asymmetric"):
        label += "-KIVI"
    return label


def kv_quant_sort_key(label: str) -> int:
    """Sort order for KV-quant labels."""
    order = {"FP16": 0, "INT8": 1, "INT4": 2, "INT2": 3, "INT2-KIVI": 4}
    return order.get(label, 99)


def model_short_name(model: str) -> str:
    """Short display name for model."""
    names = {
        "mistralai/Mistral-7B-v0.1": "Mistral-7B",
        "01-ai/Yi-1.5-9B": "Yi-1.5-9B",
        "Qwen/Qwen3-8B": "Qwen3-8B",
        "Qwen/Qwen2-7B": "Qwen2-7B",
    }
    return names.get(model, model.split("/")[-1])


def build_rows(results: list[dict]) -> list[dict]:
    """Flatten all results into per-measurement rows."""
    rows = []
    for r in results:
        model = model_short_name(r["model"])
        kv = r["kv_quant"]
        kv_label = kv_quant_label(kv)

        ppl_ref = r.get("benchmarks", {}).get("perplexity", {}).get("value")
        ppl_quant = r.get("benchmarks", {}).get("perplexity_quantized", {}).get("value")
        delta_ppl = round(ppl_quant - ppl_ref, 4) if ppl_ref and ppl_quant else None

        needle = r.get("benchmarks", {}).get("needle_in_haystack", {})
        needle_score = needle.get("success_rate") if needle else None

        for m in r["measurements"]:
            rows.append({
                "model": model,
                "kv_quant": kv_label,
                "context_len": m["context_len"],
                "prefill_ms": round(m["prefill_ms"], 1),
                "prefill_tok_s": round(m["context_len"] / (m["prefill_ms"] / 1000), 0),
                "decode_tok_s": round(m["decode_tokens_per_sec"], 1),
                "kv_cache_mb": round(m["kv_cache_mb"], 1),
                "vram_peak_mb": round(m["vram_peak_mb"], 0),
                "vram_reserved_mb": round(m.get("vram_reserved_mb", 0), 0),
                "vram_overflow": m.get("vram_overflow", False),
                "ppl_ref": ppl_ref,
                "ppl_quant": ppl_quant,
                "delta_ppl": delta_ppl,
                "needle_score": needle_score,
            })
    return rows


def write_csv(rows: list[dict], path: Path):
    """Write rows to CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"CSV: {path} ({len(rows)} rows)")


def print_delta_ppl_table(rows: list[dict]):
    """Print the key Δ-PPL comparison table @ max context per model."""
    # Get max context per model
    from collections import defaultdict
    best = defaultdict(dict)  # model -> kv_quant -> delta_ppl
    for r in rows:
        model = r["model"]
        kv = r["kv_quant"]
        # Only use the largest context row
        if kv not in best[model] or r["context_len"] > best[model][kv].get("context_len", 0):
            best[model][kv] = r

    quant_levels = ["FP16", "INT8", "INT4", "INT2", "INT2-KIVI"]
    models = sorted(best.keys(), key=lambda m: ["Mistral-7B", "Yi-1.5-9B", "Qwen3-8B", "Qwen2-7B"].index(m)
                    if m in ["Mistral-7B", "Yi-1.5-9B", "Qwen3-8B", "Qwen2-7B"] else 99)

    print(f"\n{'='*90}")
    print(f"Δ-PPL vs. FP16 Baseline (@ max context = 32768)")
    print(f"{'='*90}")
    header = f"{'Model':<14}" + "".join(f"{q:>12}" for q in quant_levels)
    print(header)
    print("-" * len(header))

    for model in models:
        parts = [f"{model:<14}"]
        for q in quant_levels:
            r = best[model].get(q)
            if r is None:
                parts.append(f"{'n/a':>12}")
            elif q == "FP16":
                parts.append(f"{r.get('ppl_ref', 0):.4f}".rjust(12))
            elif r.get("delta_ppl") is not None:
                d = r["delta_ppl"]
                if abs(d) > 100:
                    parts.append(f"+{d:.0f}".rjust(12))
                else:
                    parts.append(f"{d:+.4f}".rjust(12))
            else:
                parts.append(f"{'—':>12}")
        print("".join(parts))
    print(f"{'='*90}")


def print_vram_table(rows: list[dict]):
    """Print VRAM peak comparison @ max context."""
    from collections import defaultdict
    best = defaultdict(dict)
    for r in rows:
        model = r["model"]
        kv = r["kv_quant"]
        if kv not in best[model] or r["context_len"] > best[model][kv].get("context_len", 0):
            best[model][kv] = r

    quant_levels = ["FP16", "INT8", "INT4", "INT2", "INT2-KIVI"]
    models = sorted(best.keys(), key=lambda m: ["Mistral-7B", "Yi-1.5-9B", "Qwen3-8B", "Qwen2-7B"].index(m)
                    if m in ["Mistral-7B", "Yi-1.5-9B", "Qwen3-8B", "Qwen2-7B"] else 99)

    print(f"\n{'='*90}")
    print(f"VRAM Peak (MB) @ ctx=32768 (physical limit: 32,607 MB)")
    print(f"{'='*90}")
    header = f"{'Model':<14}" + "".join(f"{q:>12}" for q in quant_levels)
    print(header)
    print("-" * len(header))

    for model in models:
        parts = [f"{model:<14}"]
        for q in quant_levels:
            r = best[model].get(q)
            if r is None:
                parts.append(f"{'n/a':>12}")
            else:
                v = int(r["vram_peak_mb"])
                marker = " ⚠" if r.get("vram_overflow") else ""
                parts.append(f"{v}{marker}".rjust(12))
        print("".join(parts))
    print(f"{'='*90}")


def generate_latex_table(rows: list[dict], path: Path):
    """Generate a LaTeX table for the thesis."""
    from collections import defaultdict
    best = defaultdict(dict)
    for r in rows:
        model = r["model"]
        kv = r["kv_quant"]
        if kv not in best[model] or r["context_len"] > best[model][kv].get("context_len", 0):
            best[model][kv] = r

    quant_levels = ["INT8", "INT4", "INT2", "INT2-KIVI"]
    models = ["Mistral-7B", "Yi-1.5-9B", "Qwen3-8B", "Qwen2-7B"]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Perplexity degradation ($\Delta$-PPL) by KV-cache quantization level at 32k context.}",
        r"\label{tab:delta_ppl_cross_model}",
        r"\begin{tabular}{l r r r r r}",
        r"\toprule",
        r"Model & PPL\textsubscript{ref} & INT8 & INT4 & INT2 & KIVI \\",
        r"\midrule",
    ]

    for model in models:
        ppl_ref = best[model].get("FP16", {}).get("ppl_ref")
        ref_str = f"{ppl_ref:.4f}" if ppl_ref else "—"
        parts = [f"{model} & {ref_str}"]
        for q in quant_levels:
            r = best[model].get(q)
            if r is None or r.get("delta_ppl") is None:
                parts.append("—")
            else:
                d = r["delta_ppl"]
                if abs(d) > 100:
                    parts.append(f"+{d:,.0f}")
                elif abs(d) > 1:
                    parts.append(f"{d:+.2f}")
                else:
                    parts.append(f"{d:+.4f}")
        lines.append(" & ".join(parts) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"LaTeX: {path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate Phase B long-context results")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "results" / "tables")
    args = parser.parse_args()

    results = load_all_results(args.results_dir)
    print(f"Loaded {len(results)} result files from {args.results_dir.name}/")

    rows = build_rows(results)
    print(f"Expanded to {len(rows)} measurement rows")

    # Write CSV
    write_csv(rows, args.output_dir / "phase_b_long_context.csv")

    # Print tables
    print_delta_ppl_table(rows)
    print_vram_table(rows)

    # Generate LaTeX
    generate_latex_table(rows, args.output_dir / "delta_ppl_cross_model.tex")


if __name__ == "__main__":
    main()
