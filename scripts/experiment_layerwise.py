#!/usr/bin/env python3
"""Layer-weise Sensitivitaet: Sagt die Key-Kurtosis eines Layers voraus, wie
empfindlich er auf Quantisierung reagiert?

Die Arbeit zeigt, dass die *modellweite* Key-Kurtosis mit der Quantisierungs-
Degradation korreliert. Dieser Test prueft den Mechanismus *innerhalb* eines
Modells: Quantisiert man nur die Layer mit der hoechsten Kurtosis, sollte der
PPL-Anstieg deutlich groesser ausfallen als bei gleich vielen Layern mit der
niedrigsten Kurtosis.

Moeglich wird das, weil Caches in transformers >= 5 aus einzelnen Layer-Objekten
bestehen: ``Cache(layers=[...])`` nimmt eine beliebige Mischung aus
``HQQQuantizedLayer`` und ``DynamicLayer`` (FP16).

Usage:
    python experiment_layerwise.py --smoke          # nur Funktionstest
    python experiment_layerwise.py --model Qwen/Qwen3-8B
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

from benchmarks.perplexity import compute_perplexity          # noqa: E402
from core.model_loader import load_model                       # noqa: E402

KV_DIST = BASE / "results/raw/kv_distributions_v2"
OUT = BASE / "results/raw/layerwise"

MODEL_FILE_HINT = {
    "Qwen/Qwen3-8B": "qwen3",
    "Qwen/Qwen2-7B": "qwen2",
    "mistralai/Mistral-7B-v0.1": "mistral",
    "01-ai/Yi-1.5-9B": "yi",
    "google/gemma-4-E4B": "gemma",
}


def layer_kurtosis(model_id: str):
    """Key-Kurtosis je Layer aus der Verteilungsanalyse."""
    hint = MODEL_FILE_HINT.get(model_id)
    for f in sorted(KV_DIST.glob("*.json")):
        if hint and hint not in f.name.lower():
            continue
        d = json.load(open(f))
        if d.get("model") != model_id:
            continue
        return [(l["layer"], l["key"]["kurtosis"]) for l in d["layers"]]
    raise SystemExit(f"Keine Verteilungsdaten fuer {model_id} in {KV_DIST}")


def make_cache_factory(config, quant_indices, nbits, axis, group_size, residual):
    """Cache, der nur die angegebenen Layer quantisiert; Rest bleibt FP16."""
    from transformers.cache_utils import Cache, DynamicLayer, HQQQuantizedLayer

    text_cfg = config.get_text_config(decoder=True)
    n_layers = text_cfg.num_hidden_layers
    quant = set(quant_indices)

    def factory():
        layers = []
        for i in range(n_layers):
            if i in quant:
                layers.append(
                    HQQQuantizedLayer(nbits, axis, axis, group_size, residual))
            else:
                layers.append(DynamicLayer())
        return Cache(layers=layers)

    return factory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--nbits", type=int, default=2)
    ap.add_argument("--axis", type=int, default=1)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--residual", type=int, default=128)
    ap.add_argument("--ppl-tokens", type=int, default=4096)
    ap.add_argument("--fraction", type=float, default=0.5,
                    help="Anteil der Layer je Gruppe (0.5 = Haelfte)")
    ap.add_argument("--smoke", action="store_true",
                    help="Nur pruefen, ob der gemischte Cache traegt")
    ap.add_argument("--mode", choices=("kurtosis", "position", "matched", "isolate"),
                    default="kurtosis",
                    help="kurtosis: hoechste vs niedrigste Kurtosis. "
                         "position: erste vs letzte Layer (Kontrolle gegen den "
                         "Stoerfaktor Netztiefe). matched: innerhalb je eines "
                         "Positionsbandes nach Kurtosis splitten, trennt beide "
                         "Faktoren")
    args = ap.parse_args()

    kurt = layer_kurtosis(args.model)
    kurt_sorted = sorted(kurt, key=lambda x: x[1])
    k = max(1, int(len(kurt) * args.fraction))
    low_idx = [i for i, _ in kurt_sorted[:k]]
    high_idx = [i for i, _ in kurt_sorted[-k:]]

    print(f"Modell: {args.model}")
    print(f"Layer gesamt: {len(kurt)}, je Gruppe: {k}")
    print(f"  niedrigste Kurtosis: {[round(v,2) for _, v in kurt_sorted[:k]][:6]} ...")
    print(f"  hoechste  Kurtosis: ... {[round(v,2) for _, v in kurt_sorted[-k:]][-6:]}")
    print()

    model, tokenizer, info = load_model(args.model, attn_backend="sdpa", device="cuda")
    model.eval()

    n = len(kurt)
    if args.mode == "kurtosis":
        conditions = {
            "top_kurtosis": high_idx,
            "bottom_kurtosis": low_idx,
            "all_layers": list(range(n)),
        }
    elif args.mode == "position":
        conditions = {
            "first_layers": list(range(k)),
            "last_layers": list(range(n - k, n)),
        }
        import statistics as _st
        kd = dict(kurt)
        print(f"Positionskontrolle: mittlere Kurtosis "
              f"erste {k} = {_st.mean(kd[i] for i in conditions['first_layers']):.2f}, "
              f"letzte {k} = {_st.mean(kd[i] for i in conditions['last_layers']):.2f}\n")
    elif args.mode == "isolate":
        # Traegt ein einzelner Extrem-Layer den gesamten Effekt?
        import statistics as _st
        kd = dict(kurt)
        ranked = sorted(range(n), key=lambda i: kd[i])
        top9 = ranked[-9:]
        peak = ranked[-1]
        rest = [i for i in top9 if i != peak]
        conditions = {
            "peak_layer_only": [peak],
            "top9_without_peak": rest,
            "top9_full": top9,
        }
        print(f"Spitzen-Layer: {peak} (kappa={kd[peak]:.2f})")
        print(f"Rest der Top-9: {sorted(rest)} "
              f"(kappa {_st.mean(kd[i] for i in rest):.2f})\n")
    else:
        # Innerhalb je eines Positionsbandes nach Kurtosis splitten.
        # Damit ist die Netztiefe konstant gehalten und nur die Kurtosis variiert.
        import statistics as _st
        kd = dict(kurt)
        half = n // 2
        conditions = {}
        for band_name, band in (("front", list(range(half))),
                                ("back", list(range(half, n)))):
            ranked = sorted(band, key=lambda i: kd[i])
            m = len(band) // 2
            lo, hi = ranked[:m], ranked[-m:]
            conditions[f"{band_name}_low_kurt"] = lo
            conditions[f"{band_name}_high_kurt"] = hi
            print(f"{band_name}: Layer {min(band)}-{max(band)} | "
                  f"low  kappa={_st.mean(kd[i] for i in lo):6.2f} (Layer {sorted(lo)}) | "
                  f"high kappa={_st.mean(kd[i] for i in hi):6.2f} (Layer {sorted(hi)})")
        print()

    if args.smoke:
        fac = make_cache_factory(model.config, high_idx, args.nbits, args.axis,
                                 args.group_size, args.residual)
        cache = fac()
        ids = tokenizer("Hallo Welt, dies ist ein Funktionstest.",
                        return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            out = model(ids, past_key_values=cache, use_cache=True)
        c = out.past_key_values
        kinds = [type(l).__name__ for l in c.layers]
        print("Smoke-Test bestanden.")
        print(f"  Layer-Typen im Cache: "
              f"{ {t: kinds.count(t) for t in set(kinds)} }")
        print(f"  Logits: {tuple(out.logits.shape)}")
        return 0

    results = {}
    t0 = time.time()
    print("PPL FP16-Referenz (kein Cache-Eingriff) ...")
    ppl_ref = compute_perplexity(model, tokenizer, dataset="wikitext2",
                                 max_tokens=args.ppl_tokens, device="cuda")
    print(f"  PPL_ref = {ppl_ref:.4f}   [{time.time()-t0:.0f}s]\n")
    results["fp16"] = ppl_ref

    for name, idx in conditions.items():
        t1 = time.time()
        print(f"PPL mit INT{args.nbits} auf {len(idx)} Layern ({name}) ...")
        fac = make_cache_factory(model.config, idx, args.nbits, args.axis,
                                 args.group_size, args.residual)
        ppl = compute_perplexity(model, tokenizer, dataset="wikitext2",
                                 max_tokens=args.ppl_tokens, device="cuda",
                                 cache_factory=fac)
        results[name] = {"ppl": ppl, "delta": ppl - ppl_ref,
                         "layers": sorted(idx)}
        print(f"  PPL = {ppl:.4f}   Delta = {ppl - ppl_ref:+.4f}   "
              f"[{time.time()-t1:.0f}s]\n")

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = OUT / f"layerwise_{args.mode}_{args.model.split('/')[-1]}_{stamp}.json"
    json.dump({"model": args.model, "mode": args.mode,
               "nbits": args.nbits, "axis": args.axis,
               "q_group_size": args.group_size, "residual_length": args.residual,
               "ppl_tokens": args.ppl_tokens, "fraction": args.fraction,
               "layer_kurtosis": {str(i): v for i, v in kurt},
               "results": results}, open(path, "w"), indent=2)

    print("=" * 62)
    print(f"{'Bedingung':<22}{'PPL':>12}{'Delta':>12}")
    print(f"{'FP16 (Referenz)':<22}{ppl_ref:>12.4f}{0.0:>12.4f}")
    for name in conditions:
        r = results[name]
        print(f"{name:<22}{r['ppl']:>12.4f}{r['delta']:>+12.4f}")
    print("=" * 62)
    print(f"Gespeichert: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
