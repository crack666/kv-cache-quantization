#!/usr/bin/env python3
"""Aggregate Phase A v3 + backend comparison results into CSV and summary tables.

Usage:
    python aggregate_phase_a.py [--output-dir ../results/tables/]
"""
import argparse
import csv
import json
from pathlib import Path

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results" / "raw"


def load_jsons(folder: Path) -> list[dict]:
    """Load all JSON result files from a folder."""
    results = []
    for p in sorted(folder.glob("*.json")):
        if p.name == "summary.json":
            continue
        results.append(json.loads(p.read_text()))
    return results


def quant_label(kv: dict) -> str:
    if not kv["enabled"]:
        return "FP16"
    return f"INT{kv['nbits']} ({kv['backend'].upper()})"


def build_rows(results: list[dict], source: str) -> list[dict]:
    """Flatten results into per-context-length rows."""
    rows = []
    for r in results:
        kv = r["kv_quant"]
        ql = quant_label(kv)
        ppl_ref = r.get("benchmarks", {}).get("perplexity", {}).get("value")
        ppl_q = r.get("benchmarks", {}).get("perplexity_quantized", {}).get("value")
        delta_ppl = round(ppl_q - ppl_ref, 4) if ppl_ref and ppl_q else None

        for m in r["measurements"]:
            rows.append({
                "source": source,
                "model": r["model"],
                "backend": r["attn_backend"],
                "kv_quant": ql,
                "ctx": m["context_len"],
                "prefill_ms": round(m["prefill_ms"], 1),
                "decode_tok_s": round(m["decode_tokens_per_sec"], 1),
                "decode_ms": round(m["decode_ms"], 0),
                "kv_cache_mb": round(m["kv_cache_mb"], 1),
                "vram_peak_mb": round(m["vram_peak_mb"], 0),
                "vram_overflow": m.get("vram_overflow", False),
                "ppl_ref": ppl_ref,
                "ppl_quant": ppl_q,
                "delta_ppl": delta_ppl,
                "overhead_pct": m.get("overhead_pct", 0),
                "timestamp": r["timestamp"],
            })
    return rows


def write_csv(rows: list[dict], path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"CSV: {path} ({len(rows)} rows)")


def print_kv_quant_table(rows: list[dict]):
    """Print KV-quantization comparison at fixed context length."""
    # Use ctx=8192 for the comparison (largest available)
    target_ctx = max(r["ctx"] for r in rows)
    filtered = [r for r in rows if r["ctx"] == target_ctx and r["source"] == "phase_a_v3"]

    print(f"\n{'='*90}")
    print(f"KV-Cache Quantization Comparison — Mistral-7B @ ctx={target_ctx}")
    print(f"{'='*90}")
    header = f"{'KV-Quant':<16} {'Prefill':>9} {'Decode':>10} {'KV':>8} {'VRAM':>8} {'PPL':>8} {'Δ-PPL':>8} {'VRAM save':>10}"
    print(header)
    print("-" * len(header))

    # Get FP16 baseline for VRAM savings
    fp16 = next((r for r in filtered if r["kv_quant"] == "FP16"), None)
    fp16_kv = fp16["kv_cache_mb"] if fp16 else 1

    for r in sorted(filtered, key=lambda x: x["kv_cache_mb"], reverse=True):
        ppl_str = f"{r['ppl_ref']:.4f}" if r["ppl_ref"] else "n/a"
        delta = f"{r['delta_ppl']:+.4f}" if r["delta_ppl"] is not None else "—"
        saving = f"{(1 - r['kv_cache_mb'] / fp16_kv) * 100:.0f}%" if fp16_kv > 0 else "—"
        overflow = " ⚠" if r.get("vram_overflow") else ""
        print(
            f"{r['kv_quant']:<16} {r['prefill_ms']:>8.1f}ms {r['decode_tok_s']:>8.1f}t/s "
            f"{r['kv_cache_mb']:>7.1f}MB {r['vram_peak_mb']:>7.0f}MB "
            f"{ppl_str:>8} {delta:>8} {saving:>10}{overflow}"
        )
    print(f"{'='*90}")


def print_backend_table(rows: list[dict]):
    """Print backend comparison at fixed context length."""
    backend_rows = [r for r in rows if r["source"] == "backend_comparison"]
    if not backend_rows:
        return

    target_ctx = max(r["ctx"] for r in backend_rows)
    filtered = [r for r in backend_rows if r["ctx"] == target_ctx]

    print(f"\n{'='*90}")
    print(f"Attention Backend Comparison — Mistral-7B @ ctx={target_ctx}")
    print(f"{'='*90}")
    header = f"{'Backend':<10} {'Prefill':>10} {'Decode':>10} {'VRAM':>8} {'Overflow':>9}"
    print(header)
    print("-" * len(header))

    for r in sorted(filtered, key=lambda x: x["prefill_ms"]):
        overflow = "⚠ YES" if r.get("vram_overflow") else "no"
        print(
            f"{r['backend']:<10} {r['prefill_ms']:>9.1f}ms {r['decode_tok_s']:>8.1f}t/s "
            f"{r['vram_peak_mb']:>7.0f}MB {overflow:>9}"
        )
    print(f"{'='*90}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate profiling results")
    parser.add_argument("--output-dir", default="../results/tables/", help="Output directory for CSV")
    args = parser.parse_args()

    all_rows = []

    # Phase A v3
    phase_a_dir = RESULTS_ROOT / "phase_a_v3"
    if phase_a_dir.exists():
        data = load_jsons(phase_a_dir)
        all_rows.extend(build_rows(data, "phase_a_v3"))
        print(f"Loaded {len(data)} results from phase_a_v3")

    # Backend comparison
    backend_dir = RESULTS_ROOT / "backend_comparison"
    if backend_dir.exists():
        data = load_jsons(backend_dir)
        all_rows.extend(build_rows(data, "backend_comparison"))
        print(f"Loaded {len(data)} results from backend_comparison")

    if not all_rows:
        print("No results found.")
        return

    # Write CSV
    out_dir = Path(args.output_dir)
    write_csv(all_rows, out_dir / "phase_a_v3_aggregated.csv")

    # Print summary tables
    print_kv_quant_table(all_rows)
    print_backend_table(all_rows)


if __name__ == "__main__":
    main()
