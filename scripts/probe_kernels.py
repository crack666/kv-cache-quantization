#!/usr/bin/env python3
"""Post-hoc SDPA-Kernel-Validierung für einen abgeschlossenen Messlauf.

Lädt jedes Modell exakt wie der Profiler (core.model_loader, sdpa, fp16)
und lässt die Kernel-Probe (core.kernel_probe) laufen.  Kernel-Eligibility
hängt von head_dim/dtype/Maske/GQA-Flag ab, nicht von der Kontextlänge —
das Ergebnis gilt daher rückwirkend für alle Kontexte des Laufs
(Validierung für results/raw/long_context_v2, Beschluss 2026-07-18).

Usage:
    python probe_kernels.py [--output ../results/raw/long_context_v2/kernel_probe_validation.json]
"""

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from core.model_loader import load_model
from core.kernel_probe import probe_sdpa_kernels

MODELS = [
    ("mistralai/Mistral-7B-v0.1", "mistral_7b"),
    ("01-ai/Yi-1.5-9B", "yi_9b"),
    ("Qwen/Qwen3-8B", "qwen3_8b"),
    ("Qwen/Qwen2-7B", "qwen2_7b"),
    ("google/gemma-4-E4B", "gemma_e4b"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=None,
                    help="HF-Modell-ID(s) für Einzel-Check; ohne Angabe läuft die volle Liste")
    ap.add_argument("--output", default="../results/raw/long_context_v2/kernel_probe_validation.json")
    ap.add_argument("--probe-len", type=int, default=1024)
    args = ap.parse_args()

    if args.model:
        models = [(m, m.split("/")[-1].lower().replace("-", "_")) for m in args.model]
    else:
        models = MODELS

    results = {}
    for model_id, tag in models:
        print(f"\n{'=' * 70}\n{tag}: {model_id}\n{'=' * 70}")
        t0 = time.time()
        model, tokenizer, info = load_model(
            model_id, attn_backend="sdpa", kv_quant=None,
            device="cuda", dtype=torch.float16,
        )
        report = probe_sdpa_kernels(model, device="cuda", probe_len=args.probe_len)
        results[tag] = {"model": model_id, **report}
        print(f"  [{tag}: {time.time() - t0:.0f}s | math_fallback={report['math_fallback']}]")

        cleanup_fn = info.get("_cleanup_fn")
        del model, tokenizer, info
        if cleanup_fn:
            cleanup_fn()
        gc.collect()
        torch.cuda.empty_cache()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Merge: bestehende Einträge anderer Modelle bleiben erhalten (Einzel-Checks
    # aktualisieren nur ihren eigenen Eintrag)
    existing = {}
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f).get("models", {})
    existing.update(results)
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "purpose": "Post-hoc-Kernel-Validierung für long_context_v2 "
                       "(Eligibility ist ctx-unabhängig, gilt rückwirkend)",
            "dtype": "float16",
            "attn_backend": "sdpa",
            "environment": {
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_properties(0).name,
            },
            "models": existing,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")

    flagged = [t for t, r in results.items() if r["math_fallback"]]
    print(f"Math-Fallback: {flagged if flagged else 'keines der Modelle'}")


if __name__ == "__main__":
    main()
