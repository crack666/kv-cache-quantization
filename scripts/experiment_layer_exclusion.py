#!/usr/bin/env python3
"""Einzelne Layer vom Quantisieren ausnehmen: Wo sitzt der Schaden?

Anlass ist das INT4-Versagen von Qwen2-7B, das keine der in der Arbeit
erhobenen Kennzahlen erklaert. Der groesste Key-Betrag des Modells (410.8)
liegt im letzten Layer (27), nicht in Layer 0 und nicht im kurtosisreichsten
Layer (19). Gemessen wird deshalb, wie stark der Schaden sinkt, wenn genau
einer dieser Layer in FP16 bleibt und alle uebrigen quantisiert werden, und
wie viel Schaden der Layer allein anrichtet.

Die drei Kandidaten werden aus der Verteilungsmessung bestimmt, nicht von
Hand gesetzt: Layer mit dem groessten Betrag, Layer mit der hoechsten
Kurtosis, Layer 0. Die Messkette ist dieselbe wie in experiment_layerwise.py.

Usage:
    python experiment_layer_exclusion.py --model Qwen/Qwen2-7B --nbits 4
"""

import argparse
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

from benchmarks.perplexity import compute_perplexity          # noqa: E402
from core.model_loader import load_model                       # noqa: E402
from experiment_layerwise import (KV_DIST, MODEL_FILE_HINT,    # noqa: E402
                                  make_cache_factory)

OUT = BASE / "results/raw/layer_exclusion"


def layer_stats(model_id):
    """Je Layer: Kurtosis und groesster Key-Betrag aus der Verteilungsmessung."""
    hint = MODEL_FILE_HINT.get(model_id)
    for f in sorted(KV_DIST.glob("*.json")):
        if hint and hint not in f.name.lower():
            continue
        d = json.load(open(f, encoding="utf-8"))
        if d.get("model") == model_id:
            return {l["layer"]: (l["key"]["kurtosis"],
                                 max(abs(l["key"]["min"]), abs(l["key"]["max"])))
                    for l in d["layers"]}
    raise SystemExit(f"Keine Verteilungsdaten fuer {model_id} in {KV_DIST}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-7B")
    ap.add_argument("--nbits", type=int, default=4)
    ap.add_argument("--axis", type=int, default=1)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--residual", type=int, default=128)
    ap.add_argument("--ppl-tokens", type=int, default=4096)
    ap.add_argument("--conditions", nargs="+", metavar="ART:LAYER",
                    help="statt der automatischen Kandidaten, z. B. only:0 all_but:0 "
                         "(ART ist only oder all_but)")
    args = ap.parse_args()

    stats = layer_stats(args.model)
    n = len(stats)
    peak_abs = max(stats, key=lambda i: stats[i][1])
    peak_kurt = max(stats, key=lambda i: stats[i][0])
    candidates = {"max_abs": peak_abs, "max_kurtosis": peak_kurt, "layer0": 0}

    print(f"Modell: {args.model}, {n} Layer, INT{args.nbits}, axis {args.axis}")
    for name, i in candidates.items():
        print(f"  {name:<13} Layer {i:>2}: kappa {stats[i][0]:7.2f}, max|K| {stats[i][1]:7.1f}")
    print()

    conditions = {"all_layers": list(range(n))}
    if args.conditions:
        for spec in args.conditions:
            kind, layer = spec.split(":")
            layer = int(layer)
            if kind == "only":
                conditions[f"only_{layer}"] = [layer]
            elif kind == "all_but":
                conditions[f"all_but_{layer}"] = [i for i in range(n) if i != layer]
            else:
                raise SystemExit(f"unbekannte Bedingung: {spec}")
    else:
        for layer in sorted(set(candidates.values())):
            conditions[f"all_but_{layer}"] = [i for i in range(n) if i != layer]
        conditions[f"only_{peak_abs}"] = [peak_abs]

    model, tokenizer, _ = load_model(args.model, attn_backend="sdpa", device="cuda")
    model.eval()

    t0 = time.time()
    print("PPL FP16-Referenz ...")
    ppl_ref = compute_perplexity(model, tokenizer, dataset="wikitext2",
                                 max_tokens=args.ppl_tokens, device="cuda")
    print(f"  PPL_ref = {ppl_ref:.4f}   [{time.time()-t0:.0f}s]\n")
    results = {"fp16": ppl_ref}

    for name, idx in conditions.items():
        t1 = time.time()
        print(f"PPL mit INT{args.nbits} auf {len(idx)} Layern ({name}) ...")
        fac = make_cache_factory(model.config, idx, args.nbits, args.axis,
                                 args.group_size, args.residual)
        ppl = compute_perplexity(model, tokenizer, dataset="wikitext2",
                                 max_tokens=args.ppl_tokens, device="cuda",
                                 cache_factory=fac)
        results[name] = {"ppl": ppl, "delta": ppl - ppl_ref, "layers": sorted(idx)}
        print(f"  PPL = {ppl:.4f}   Delta = {ppl - ppl_ref:+.4f}   [{time.time()-t1:.0f}s]\n")

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = OUT / f"layer_exclusion_{args.model.split('/')[-1]}_int{args.nbits}_{stamp}.json"
    json.dump({"model": args.model, "nbits": args.nbits, "axis": args.axis,
               "q_group_size": args.group_size, "residual_length": args.residual,
               "ppl_tokens": args.ppl_tokens, "candidates": candidates,
               "layer_stats": {str(i): {"kurtosis": k, "max_abs": a}
                               for i, (k, a) in stats.items()},
               "results": results}, open(path, "w", encoding="utf-8"), indent=2)

    print("=" * 62)
    print(f"{'Bedingung':<22}{'PPL':>12}{'Delta':>14}")
    print(f"{'FP16 (Referenz)':<22}{ppl_ref:>12.4f}{0.0:>14.4f}")
    for name in conditions:
        r = results[name]
        print(f"{name:<22}{r['ppl']:>12.4f}{r['delta']:>+14.4f}")
    print("=" * 62)
    print(f"Gespeichert: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
