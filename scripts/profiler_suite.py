#!/usr/bin/env python3
"""Profiler Suite — unified CLI for multi-dimensional LLM benchmarking.

Runs the cartesian product of:
    Model × Attention-Backend × KV-Quant × Context-Lengths
and optionally evaluates benchmarks (PPL, MMLU, …) for each combination.

Produces one JSON v2 result file per combination under ``--output-dir``.

Examples:
    # Minimal: profile one model with FP16 baseline
    python profiler_suite.py --model gpt2 --context-lengths 128 256

    # WisSem: compare attention backends
    python profiler_suite.py --model mistralai/Mistral-7B-v0.3 \\
        --attn-backend sdpa eager --benchmarks ppl

    # MA: full matrix with KV-quant
    python profiler_suite.py --model mistralai/Mistral-7B-v0.3 \\
        --attn-backend sdpa --kv-quant none int8-hqq int4-hqq \\
        --context-lengths 1024 4096 8192 16384 \\
        --benchmarks ppl mmlu --output-dir results/raw/
"""

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import torch

# Ensure the scripts package is importable when invoked directly
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from core.model_loader import load_model, _collect_environment
from core.kv_cache import (
    measure_kv_cache_size,
    patch_quantized_cache,
    reset_timings,
    get_timings,
)
from core.vram_profiler import VRAMProfiler
from core.metrics import measure_prefill_latency, measure_decode_throughput


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Profiler Suite — multi-dimensional LLM benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model
    p.add_argument("--model", required=True, help="HF model id or local path")

    # Dimensions
    p.add_argument(
        "--attn-backend",
        nargs="+",
        default=["sdpa"],
        choices=["sdpa", "eager", "flash_attention_2", "sage"],
        help="Attention backend(s) to test (default: sdpa)",
    )
    p.add_argument(
        "--kv-quant",
        nargs="+",
        default=["none"],
        help="KV-cache quantization spec(s). Format: int{2,4,8}-{hqq,quanto}[-kivi]. "
             "The '-kivi' suffix enables asymmetric axes (keys per-channel, values per-token). "
             "Examples: none, int8-hqq, int4-hqq, int4-hqq-kivi, int2-hqq-kivi (default: none)",
    )
    p.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[1024, 4096, 8192, 16384],
        help="Context lengths to profile (default: 1024 4096 8192 16384)",
    )

    # Benchmarks
    p.add_argument(
        "--benchmarks",
        nargs="+",
        default=[],
        choices=["ppl", "mmlu", "hellaswag"],
        help="Benchmarks to run after profiling (default: none)",
    )
    p.add_argument("--ppl-dataset", default="wikitext2", choices=["wikitext2", "pg19"])
    p.add_argument("--ppl-tokens", type=int, default=4096, help="Sliding-window size for PPL")

    # Measurement
    p.add_argument("--warmup-runs", type=int, default=2)
    p.add_argument("--measure-runs", type=int, default=5)
    p.add_argument("--measure-power", action="store_true", help="Enable GPU power sampling")
    p.add_argument("--decode-tokens", type=int, default=128, help="Tokens to generate for decode throughput")

    # Output
    p.add_argument("--output-dir", default="results/raw/", help="Output directory for JSON v2 files")
    p.add_argument("--summary-file", default=None,
                   help="Path for a compact JSON summary of all combinations (agent-friendly)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    return p


# ═══════════════════════════════════════════════════════════════════════════
# Single-combination runner
# ═══════════════════════════════════════════════════════════════════════════

def run_single_combination(
    model_id: str,
    attn_backend: str,
    kv_quant: str,
    context_lengths: list,
    args,
) -> dict:
    """Profile one (attn_backend, kv_quant) combination across all context lengths."""
    combo_t0 = time.time()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # VRAM profiler — init BEFORE model loading to capture model VRAM
    profiler = None
    vram_total_mb = 0.0
    if args.device == "cuda":
        profiler = VRAMProfiler()
        vram_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)

    # Load model
    model, tokenizer, info = load_model(
        model_id,
        attn_backend=attn_backend,
        kv_quant=kv_quant if kv_quant != "none" else None,
        device=args.device,
        dtype=torch.float16,
    )

    kv_cfg = info["kv_quant"]
    if kv_cfg["enabled"]:
        patch_quantized_cache()
        mode = "KIVI asymmetric" if kv_cfg["asymmetric"] else "symmetric"
        print(f"  KV-Quant: int{kv_cfg['nbits']}-{kv_cfg['backend']} ({mode}, axis_key={kv_cfg['axis_key']}, axis_value={kv_cfg['axis_value']})")

    if profiler:
        profiler.log_vram("model_loaded")

    # Power sampler (lazy import)
    power_ctx = None
    if args.measure_power and args.device == "cuda":
        from core.power_sampler import PowerSampler
        power_ctx = PowerSampler(handle=profiler.handle if profiler else None)

    # ── Profiling measurements per context length ────────────────────────
    measurements = []

    for ctx_len in context_lengths:
        print(f"\n--- Context: {ctx_len} ---")
        ctx_t0 = time.time()

        # Build input
        input_text = "Hello world " * (ctx_len // 2)
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=ctx_len)
        input_ids = inputs["input_ids"].to(args.device)
        actual_len = input_ids.shape[-1]
        print(f"  Input tokens: {actual_len}")

        # Prepare cache
        if kv_cfg["enabled"]:
            from transformers import QuantizedCache
            cache = QuantizedCache(
                backend=kv_cfg["backend"],
                config=info["text_config"],
                nbits=kv_cfg["nbits"],
                axis_key=kv_cfg["axis_key"],
                axis_value=kv_cfg["axis_value"],
            )
        else:
            from transformers import DynamicCache
            cache = DynamicCache()

        # Clear state
        if args.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        reset_timings()

        # Start power sampling
        if power_ctx:
            power_ctx.start()

        # Prefill
        prefill = measure_prefill_latency(
            model, input_ids, past_key_values=cache, warmup_runs=args.warmup_runs
        )
        filled_cache = prefill["past_key_values"]

        # KV-cache size
        kv_mb, kv_type = measure_kv_cache_size(filled_cache)

        # Decode throughput
        # Use last token as decode prompt
        last_token = input_ids[:, -1:]

        # Build a prefill_fn for caches that can't be deepcopied (e.g. quanto)
        def _re_prefill(_ids=input_ids, _kv_cfg=kv_cfg, _model=model, _tcfg=info["text_config"]):
            if _kv_cfg["enabled"]:
                from transformers import QuantizedCache
                c = QuantizedCache(
                    backend=_kv_cfg["backend"],
                    config=_tcfg,
                    nbits=_kv_cfg["nbits"],
                    axis_key=_kv_cfg["axis_key"],
                    axis_value=_kv_cfg["axis_value"],
                )
            else:
                from transformers import DynamicCache
                c = DynamicCache()
            with torch.no_grad():
                out = _model(_ids, past_key_values=c, use_cache=True)
            return out.past_key_values

        decode = measure_decode_throughput(
            model,
            last_token,
            n_tokens=args.decode_tokens,
            past_key_values=filled_cache,
            warmup_runs=args.warmup_runs,
            prefill_fn=_re_prefill,
        )

        # VRAM peak + overflow detection
        vram_peak_mb = 0.0
        vram_overflow = False
        if args.device == "cuda":
            vram_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            if vram_total_mb > 0 and vram_peak_mb > vram_total_mb:
                vram_overflow = True
                overflow_mb = vram_peak_mb - vram_total_mb
                print(f"  ⚠ VRAM OVERFLOW: peak {vram_peak_mb:.0f} MB > physical {vram_total_mb:.0f} MB "
                      f"(+{overflow_mb:.0f} MB spilled to system RAM via PCIe — results may be unreliable)")

        # Power
        power_stats = {"avg_watts": 0.0}
        if power_ctx:
            power_stats = power_ctx.stop()

        # Quant overhead
        timing_summary = get_timings().summary()

        # Energy per token
        total_decode_s = decode["decode_ms"] / 1000
        energy_mj = 0.0
        if power_stats["avg_watts"] > 0 and decode["tokens"] > 0:
            energy_mj = (power_stats["avg_watts"] * total_decode_s * 1000) / decode["tokens"]

        # Compute overhead percentage
        overhead_pct = 0.0
        total_ms = prefill["prefill_ms"] + decode["decode_ms"]
        if total_ms > 0:
            overhead_pct = (timing_summary["total_overhead_ms"] / total_ms) * 100

        m = {
            "context_len": actual_len,
            "prefill_ms": prefill["prefill_ms"],
            "prefill_tokens_per_sec": prefill["tokens_per_sec"],
            "decode_ms": decode["decode_ms"],
            "decode_tokens": decode["tokens"],
            "decode_tokens_per_sec": decode["tokens_per_sec"],
            "vram_peak_mb": round(vram_peak_mb, 1),
            "kv_cache_mb": round(kv_mb, 2),
            "kv_cache_type": kv_type,
            "quant_overhead_ms": timing_summary["quantize_total_ms"],
            "dequant_overhead_ms": timing_summary["dequantize_total_ms"],
            "overhead_pct": round(overhead_pct, 2),
            "avg_power_watts": round(power_stats["avg_watts"], 1),
            "energy_mj_per_token": round(energy_mj, 2),
            "ctx_elapsed_s": round(time.time() - ctx_t0, 1),
            "vram_overflow": vram_overflow,
        }
        measurements.append(m)
        ctx_elapsed = time.time() - ctx_t0
        print(f"  Prefill: {m['prefill_ms']:.1f}ms ({m['prefill_tokens_per_sec']:.0f} tok/s)"
              f" | Decode: {m['decode_ms']:.0f}ms total, {m['decode_tokens_per_sec']:.1f} tok/s ({m['decode_tokens']} new tokens)"
              f" | KV: {m['kv_cache_mb']:.1f}MB | VRAM peak: {m['vram_peak_mb']:.0f}MB"
              f" | wall: {ctx_elapsed:.1f}s")

        # Free cache for next iteration
        del filled_cache, cache
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # ── Benchmarks ───────────────────────────────────────────────────────
    benchmarks = {}

    if "ppl" in args.benchmarks:
        from benchmarks.perplexity import compute_perplexity

        # Reference PPL (no cache — intrinsic model quality)
        print("\nRunning perplexity benchmark (reference, no cache)...")
        ppl_ref = compute_perplexity(
            model, tokenizer,
            dataset=args.ppl_dataset,
            max_tokens=args.ppl_tokens,
            device=args.device,
        )
        benchmarks["perplexity"] = {
            "dataset": args.ppl_dataset,
            "tokens": args.ppl_tokens,
            "value": ppl_ref,
        }
        print(f"  PPL ref ({args.ppl_dataset}): {ppl_ref:.4f}")

        # Quantized PPL (through the cache — measures quant quality loss)
        if kv_cfg["enabled"]:
            def _make_cache(_tcfg=info["text_config"]):
                from transformers import QuantizedCache
                return QuantizedCache(
                    backend=kv_cfg["backend"],
                    config=_tcfg,
                    nbits=kv_cfg["nbits"],
                    axis_key=kv_cfg["axis_key"],
                    axis_value=kv_cfg["axis_value"],
                )

            print("  Running perplexity benchmark (quantized cache)...")
            ppl_quant = compute_perplexity(
                model, tokenizer,
                dataset=args.ppl_dataset,
                max_tokens=args.ppl_tokens,
                device=args.device,
                cache_factory=_make_cache,
            )
            benchmarks["perplexity_quantized"] = {
                "dataset": args.ppl_dataset,
                "tokens": args.ppl_tokens,
                "value": ppl_quant,
            }
            delta = ppl_quant - ppl_ref
            print(f"  PPL quant ({args.ppl_dataset}): {ppl_quant:.4f}  (Δ={delta:+.4f})")
        print(f"  PPL ({args.ppl_dataset}): {ppl_ref:.4f}")

    lm_eval_tasks = [t for t in args.benchmarks if t in ("mmlu", "hellaswag")]
    if lm_eval_tasks:
        print(f"\nRunning lm-eval tasks: {lm_eval_tasks}")
        from benchmarks.lm_eval_wrapper import run_lm_eval
        eval_results = run_lm_eval(
            model, tokenizer,
            tasks=lm_eval_tasks,
            device=args.device,
            kv_quant_config=kv_cfg if kv_cfg["enabled"] else None,
        )
        for task_name, scores in eval_results.items():
            benchmarks[task_name] = scores
            print(f"  {task_name}: acc={scores['accuracy']:.4f} ± {scores['stderr']:.4f}")

    # ── Assemble JSON v2 ─────────────────────────────────────────────────
    timestamp = datetime.now()
    model_short = model_id.split("/")[-1].lower().replace("-", "_")
    quant_tag = kv_quant.replace("-", "_") if kv_quant != "none" else "fp16"
    experiment_id = f"{model_short}_{attn_backend}_{quant_tag}_{timestamp.strftime('%Y%m%d_%H%M%S')}"

    combo_elapsed_s = round(time.time() - combo_t0, 1)

    result = {
        "schema_version": "2.0",
        "experiment_id": experiment_id,
        "timestamp": timestamp.isoformat(timespec="seconds"),
        "model": model_id,
        "model_config": info["model_config"],
        "attn_backend": attn_backend,
        "kv_quant": kv_cfg,
        "environment": info["environment"],
        "hardware": {
            "vram_total_mb": round(vram_total_mb, 1),
            "gpu_name": torch.cuda.get_device_properties(0).name if args.device == "cuda" else "n/a",
        },
        "config": {
            "seed": args.seed,
            "warmup_runs": args.warmup_runs,
            "measure_runs": args.measure_runs,
            "decode_tokens": args.decode_tokens,
            "ppl_dataset": args.ppl_dataset,
            "ppl_tokens": args.ppl_tokens,
        },
        "measurements": measurements,
        "benchmarks": benchmarks,
        "combo_elapsed_s": combo_elapsed_s,
    }

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{experiment_id}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Cleanup
    cleanup_fn = info.get("_cleanup_fn")
    del model, tokenizer
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    if cleanup_fn:
        cleanup_fn()

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = build_parser()
    args = parser.parse_args()

    combos = list(product(args.attn_backend, args.kv_quant))
    total = len(combos)
    suite_start = time.time()

    print("=" * 80)
    print(f"Profiler Suite — {total} combination(s)")
    print(f"  Model:    {args.model}")
    print(f"  Backends: {args.attn_backend}")
    print(f"  KV-Quant: {args.kv_quant}")
    print(f"  Contexts: {args.context_lengths}")
    print(f"  Benchmarks: {args.benchmarks or '(profiling only)'}")
    print("=" * 80)

    results = []
    for idx, (backend, quant) in enumerate(combos, 1):
        print(f"\n{'='*80}")
        print(f"[{idx}/{total}] attn={backend} | kv_quant={quant}")
        print(f"{'='*80}")

        try:
            r = run_single_combination(
                args.model, backend, quant, args.context_lengths, args
            )
            results.append(r)

            # Running ETA
            elapsed = time.time() - suite_start
            avg_per_combo = elapsed / idx
            remaining = avg_per_combo * (total - idx)
            rem_min, rem_sec = divmod(remaining, 60)
            print(f"  [{idx}/{total}] combo took {r['combo_elapsed_s']:.0f}s | "
                  f"elapsed {elapsed:.0f}s | ETA ~{int(rem_min)}m{rem_sec:.0f}s")
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM — skipping {backend}/{quant}")
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    suite_elapsed = time.time() - suite_start
    suite_min, suite_sec = divmod(suite_elapsed, 60)

    # ── Compact summary table (agent-friendly) ───────────────────────────
    summary_rows = []
    for r in results:
        # Extract key metrics from the last (longest) context measurement
        last_m = r["measurements"][-1] if r["measurements"] else {}
        ppl_ref = r.get("benchmarks", {}).get("perplexity", {}).get("value")
        ppl_quant = r.get("benchmarks", {}).get("perplexity_quantized", {}).get("value")
        ppl_delta = round(ppl_quant - ppl_ref, 4) if ppl_ref and ppl_quant else None

        row = {
            "backend": r["attn_backend"],
            "kv_quant": "fp16" if not r["kv_quant"]["enabled"] else f"int{r['kv_quant']['nbits']}-{r['kv_quant']['backend']}{'(kivi)' if r['kv_quant'].get('asymmetric') else ''}",
            "axis_key": r["kv_quant"].get("axis_key"),
            "axis_value": r["kv_quant"].get("axis_value"),
            "asymmetric": r["kv_quant"].get("asymmetric", False),
            "ctx": last_m.get("context_len", "?"),
            "prefill_ms": last_m.get("prefill_ms"),
            "decode_tok_s": last_m.get("decode_tokens_per_sec"),
            "kv_mb": last_m.get("kv_cache_mb"),
            "vram_peak_mb": last_m.get("vram_peak_mb"),
            "vram_overflow": any(m.get("vram_overflow", False) for m in r["measurements"]),
            "ppl": ppl_ref,
            "ppl_quant": ppl_quant,
            "ppl_delta": ppl_delta,
            "combo_elapsed_s": r.get("combo_elapsed_s", 0),
            "json_file": r["experiment_id"] + ".json",
        }
        summary_rows.append(row)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY — {len(results)}/{total} combinations | {int(suite_min)}m {suite_sec:.1f}s")
    print(f"{'='*80}")
    # Header
    header = f"{'Backend':<8} {'KV-Quant':<14} {'Ctx':>5} {'Prefill':>9} {'Decode':>10} {'KV':>8} {'VRAM':>8} {'PPL':>8} {'Δ-PPL':>8} {'Time':>6}"
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        ppl_str = f"{row['ppl']:.4f}" if row['ppl'] else "n/a"
        delta_str = f"{row['ppl_delta']:+.4f}" if row['ppl_delta'] is not None else "—"
        time_str = f"{row['combo_elapsed_s']:.0f}s"
        overflow_marker = " ⚠️" if row.get("vram_overflow") else ""
        print(
            f"{row['backend']:<8} {row['kv_quant']:<14} {row['ctx']:>5} "
            f"{row['prefill_ms']:>8.1f}ms {row['decode_tok_s']:>8.1f}t/s "
            f"{row['kv_mb']:>7.0f}MB {row['vram_peak_mb']:>7.0f}MB "
            f"{ppl_str:>8} {delta_str:>8} {time_str:>6}{overflow_marker}"
        )
    print(f"{'='*80}")

    # Write summary file if requested
    if args.summary_file:
        summary_path = Path(args.summary_file)
        # Avoid overwriting: insert timestamp before extension if file exists
        if summary_path.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_path = summary_path.with_stem(f"{summary_path.stem}_{ts}")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "model": args.model,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "total_runtime_s": round(suite_elapsed, 1),
            "combinations": summary_rows,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary: {summary_path}")

    print(f"Done. {len(results)}/{total} combinations completed.")


if __name__ == "__main__":
    main()
