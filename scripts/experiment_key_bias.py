#!/usr/bin/env python3
"""Mechanismustest: Ist der Key-Bias von Layer 0 die Ursache des Qwen2-Versagens?

experiment_layer_exclusion.py hat das INT4-Versagen von Qwen2-7B auf Layer 0
eingegrenzt. Dort liegen die extremen Key-Werte (-161 bis 167) fast exakt auf
dem Bias der Key-Projektion (bis 166). Die grossen Bias-Kanaele gehoeren zu den
niedrigsten RoPE-Frequenzen und sind ueber die Tokens nahezu konstant. Ein
konstanter Key-Anteil verschiebt je Query alle Logits gleich und aendert die
Softmax kaum - er bestimmt aber bei per-Token-Quantisierung die Schrittweite
jeder Gruppe und loescht so die kleinen, informativen Key-Anteile.

Vorhersage, falls das stimmt: Bias entfernen aendert die FP16-Perplexitaet
kaum, laesst aber den INT4-Schaden weitgehend verschwinden.

Usage:
    python experiment_key_bias.py --model Qwen/Qwen2-7B --nbits 4
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
from experiment_layerwise import make_cache_factory            # noqa: E402

OUT = BASE / "results/raw/key_bias"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-7B")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=50.0,
                    help="Bias-Kanaele mit |b| ueber diesem Wert gelten als gross")
    ap.add_argument("--nbits", type=int, default=4)
    ap.add_argument("--axis", type=int, default=1)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--residual", type=int, default=128)
    ap.add_argument("--ppl-tokens", type=int, default=4096)
    args = ap.parse_args()

    model, tokenizer, _ = load_model(args.model, attn_backend="sdpa", device="cuda")
    model.eval()
    k_proj = model.model.layers[args.layer].self_attn.k_proj
    if k_proj.bias is None:
        raise SystemExit(f"{args.model} hat in Layer {args.layer} keinen Key-Bias")
    original = k_proj.bias.detach().clone()
    big = (original.float().abs() > args.threshold).nonzero().flatten().tolist()
    print(f"Modell {args.model}, Layer {args.layer}: {len(big)} von {original.numel()} "
          f"Bias-Kanaelen mit |b| > {args.threshold}: {big}")
    print(f"  max|b| = {original.float().abs().max():.1f}\n")

    n = model.config.get_text_config(decoder=True).num_hidden_layers
    quant_all = make_cache_factory(model.config, list(range(n)), args.nbits, args.axis,
                                   args.group_size, args.residual)

    def set_bias(variant):
        b = original.clone()
        if variant == "zero_big":
            b[big] = 0
        elif variant == "zero_all":
            b.zero_()
        with torch.no_grad():
            k_proj.bias.copy_(b)

    conditions = [
        ("fp16_original", "original", None),
        (f"int{args.nbits}_original", "original", quant_all),
        ("fp16_zero_big", "zero_big", None),
        (f"int{args.nbits}_zero_big", "zero_big", quant_all),
        ("fp16_zero_all", "zero_all", None),
        (f"int{args.nbits}_zero_all", "zero_all", quant_all),
    ]
    results = {}
    for name, variant, fac in conditions:
        set_bias(variant)
        t1 = time.time()
        kwargs = {"cache_factory": fac} if fac else {}
        ppl = compute_perplexity(model, tokenizer, dataset="wikitext2",
                                 max_tokens=args.ppl_tokens, device="cuda", **kwargs)
        results[name] = ppl
        print(f"  {name:<22} PPL = {ppl:.4f}   [{time.time()-t1:.0f}s]")
    set_bias("original")

    ref = results["fp16_original"]
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = OUT / f"key_bias_{args.model.split('/')[-1]}_L{args.layer}_int{args.nbits}_{stamp}.json"
    json.dump({"model": args.model, "layer": args.layer, "threshold": args.threshold,
               "big_channels": big,
               "big_values": [round(original[i].item(), 2) for i in big],
               "nbits": args.nbits, "axis": args.axis,
               "q_group_size": args.group_size, "residual_length": args.residual,
               "ppl_tokens": args.ppl_tokens,
               "results": {k: {"ppl": v, "delta": v - ref} for k, v in results.items()}},
              open(path, "w", encoding="utf-8"), indent=2)

    print("\n" + "=" * 52)
    print(f"{'Bedingung':<24}{'PPL':>12}{'Delta':>14}")
    for name, v in results.items():
        print(f"{name:<24}{v:>12.4f}{v - ref:>+14.4f}")
    print("=" * 52)
    print(f"Gespeichert: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
