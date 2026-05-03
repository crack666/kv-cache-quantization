#!/usr/bin/env python3
"""Merge individual combo JSONs into a unified summary JSON.

Reads all per-combo result JSONs for a given model from a results directory
and produces a single summary JSON in the same format as profiler_suite.py's
--summary-file output.

Usage:
    # Merge all Qwen3-8B results from long_context/
    python merge_summary.py ../results/raw/long_context/ \
        --model Qwen/Qwen3-8B \
        --output ../results/raw/long_context/qwen3_8b_summary.json

    # Dry-run: just print what would be merged
    python merge_summary.py ../results/raw/long_context/ \
        --model Qwen/Qwen3-8B --dry-run

    # Auto-detect model from filenames (uses most common model in folder)
    python merge_summary.py ../results/raw/long_context/ --auto
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def load_combo_jsons(folder: Path, model_filter: str | None = None) -> list[dict]:
    """Load all individual combo JSONs from folder, optionally filtered by model."""
    results = []
    for p in sorted(folder.glob("*.json")):
        # Skip summary files and backups
        if "summary" in p.name or p.suffix != ".json" or p.name.endswith(".bak"):
            continue
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARN: skipping {p.name} — {e}")
            continue

        if model_filter and data.get("model") != model_filter:
            continue
        data["_source_file"] = p.name
        results.append(data)
    return results


def make_combo_key(r: dict) -> str:
    """Create a unique key for a combo (backend + kv_quant config)."""
    kv = r["kv_quant"]
    if not kv["enabled"]:
        quant_str = "fp16"
    else:
        quant_str = f"int{kv['nbits']}-{kv['backend']}"
        if kv.get("asymmetric"):
            quant_str += "-kivi"
    return f"{r['attn_backend']}|{quant_str}"


def make_summary_row(r: dict) -> dict:
    """Build a summary row from a combo result (same format as profiler_suite.py)."""
    kv = r["kv_quant"]
    if not kv["enabled"]:
        kv_label = "fp16"
    else:
        kv_label = f"int{kv['nbits']}-{kv['backend']}"
        if kv.get("asymmetric"):
            kv_label += "(kivi)"

    last_m = r["measurements"][-1] if r["measurements"] else {}
    ppl_ref = r.get("benchmarks", {}).get("perplexity", {}).get("value")
    ppl_quant = r.get("benchmarks", {}).get("perplexity_quantized", {}).get("value")
    ppl_delta = round(ppl_quant - ppl_ref, 4) if ppl_ref and ppl_quant else None

    return {
        "backend": r["attn_backend"],
        "kv_quant": kv_label,
        "axis_key": kv.get("axis_key"),
        "axis_value": kv.get("axis_value"),
        "asymmetric": kv.get("asymmetric", False),
        "ctx": last_m.get("context_len", "?"),
        "prefill_ms": last_m.get("prefill_ms"),
        "decode_tok_s": last_m.get("decode_tokens_per_sec"),
        "kv_mb": last_m.get("kv_cache_mb"),
        "vram_peak_mb": last_m.get("vram_peak_mb"),
        "vram_reserved_mb": last_m.get("vram_reserved_mb"),
        "vram_overflow": any(m.get("vram_overflow", False) for m in r["measurements"]),
        "ppl": ppl_ref,
        "ppl_quant": ppl_quant,
        "ppl_delta": ppl_delta,
        "combo_elapsed_s": r.get("combo_elapsed_s", 0),
        "json_file": r.get("_source_file", r.get("experiment_id", "unknown") + ".json"),
    }


def print_summary_table(rows: list[dict]):
    """Print a compact summary table."""
    header = (f"{'Backend':<8} {'KV-Quant':<16} {'Ctx':>5} {'Prefill':>9} "
              f"{'Decode':>10} {'KV':>8} {'VRAM':>8} {'PPL':>8} {'Δ-PPL':>8}")
    print(header)
    print("-" * len(header))
    for row in rows:
        ppl_str = f"{row['ppl']:.4f}" if row['ppl'] else "n/a"
        delta_str = f"{row['ppl_delta']:+.4f}" if row['ppl_delta'] is not None else "—"
        overflow = " ⚠️" if row.get("vram_overflow") else ""
        print(
            f"{row['backend']:<8} {row['kv_quant']:<16} {row['ctx']:>5} "
            f"{row['prefill_ms']:>8.1f}ms {row['decode_tok_s']:>8.1f}t/s "
            f"{row['kv_mb']:>7.0f}MB {row['vram_peak_mb']:>7.0f}MB "
            f"{ppl_str:>8} {delta_str:>8}{overflow}"
        )


def main():
    p = argparse.ArgumentParser(description="Merge individual combo JSONs into a unified summary")
    p.add_argument("folder", type=Path, help="Directory containing per-combo JSON files")
    p.add_argument("--model", type=str, help="Model name to filter (e.g. 'Qwen/Qwen3-8B')")
    p.add_argument("--auto", action="store_true", help="Auto-detect model from filenames")
    p.add_argument("--output", "-o", type=Path, help="Output path for merged summary JSON")
    p.add_argument("--dry-run", action="store_true", help="Print what would be merged without writing")
    p.add_argument("--prefer-latest", action="store_true", default=True,
                   help="When duplicate combos exist, keep the latest (default: True)")
    args = p.parse_args()

    if not args.folder.exists():
        print(f"ERROR: folder not found: {args.folder}")
        sys.exit(1)

    # Load all JSONs (optionally filtered)
    all_results = load_combo_jsons(args.folder, model_filter=args.model)
    if not all_results:
        if args.model:
            print(f"No results found for model '{args.model}' in {args.folder}")
        else:
            print(f"No result JSONs found in {args.folder}")
        sys.exit(1)

    # Auto-detect model if needed
    if args.auto and not args.model:
        from collections import Counter
        models = Counter(r["model"] for r in all_results)
        if len(models) > 1:
            print(f"Multiple models found: {dict(models)}")
            print("Use --model to specify which one, or results will be grouped per-model")
        args.model = models.most_common(1)[0][0]
        all_results = [r for r in all_results if r["model"] == args.model]
        print(f"Auto-detected model: {args.model}")

    # Deduplicate: group by combo key, keep latest
    by_combo: dict[str, list[dict]] = {}
    for r in all_results:
        key = make_combo_key(r)
        by_combo.setdefault(key, []).append(r)

    final_results = []
    for key, variants in sorted(by_combo.items()):
        if len(variants) > 1:
            variants.sort(key=lambda r: r.get("timestamp", ""))
            if args.prefer_latest:
                chosen = variants[-1]
                skipped = [v["_source_file"] for v in variants[:-1]]
                print(f"  DEDUP: {key} — keeping {chosen['_source_file']}, skipping {skipped}")
            else:
                chosen = variants[0]
        else:
            chosen = variants[0]
        final_results.append(chosen)

    # Build summary rows
    summary_rows = [make_summary_row(r) for r in final_results]
    total_runtime = sum(r.get("combo_elapsed_s", 0) for r in final_results)

    # Print
    model_name = args.model or final_results[0]["model"]
    print(f"\n{'='*80}")
    print(f"MERGED SUMMARY — {model_name} — {len(summary_rows)} combinations")
    print(f"{'='*80}")
    print_summary_table(summary_rows)
    print(f"{'='*80}")
    print(f"Source files: {[r['json_file'] for r in summary_rows]}")

    if args.dry_run:
        print("\n(dry-run — no file written)")
        return

    # Write merged summary
    if not args.output:
        print("\nNo --output specified. Use -o to write the merged summary.")
        return

    output_path = args.output
    if output_path.exists():
        # Backup existing
        bak = output_path.with_suffix(".json.bak")
        output_path.rename(bak)
        print(f"  Backed up existing summary to {bak.name}")

    summary = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "total_runtime_s": round(total_runtime, 1),
        "merged_from": [r["json_file"] for r in summary_rows],
        "combinations": summary_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWritten: {output_path} ({len(summary_rows)} combinations)")


if __name__ == "__main__":
    main()
