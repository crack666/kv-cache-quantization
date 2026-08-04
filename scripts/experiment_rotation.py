#!/usr/bin/env python3
"""Kausaler Test: Senkt eine orthogonale Rotation der Keys die Kurtosis, und
rettet das ein Modell, das unter Quantisierung zusammenbricht?

Die Arbeit zeigt eine Korrelation zwischen Key-Kurtosis und Quantisierungs-
Degradation. Dieser Test greift ein, statt nur zu beobachten: Eine Hadamard-
Rotation H verteilt einzelne Ausreisser ueber viele Dimensionen und senkt damit
die Kurtosis -- laesst die Attention-Scores aber unveraendert, denn

    (QH)(KH)^T = Q H H^T K^T = Q K^T     (H orthogonal)

Quantisiert wird also im rotierten Raum. Bleibt die Qualitaet erhalten, wo
unrotierte Quantisierung scheitert, ist der Zusammenhang zwischen Kurtosis und
Toleranz nicht mehr nur beobachtet, sondern hergestellt.

Bedingungen:
    fp16              Referenz
    quant_only        Quantisierung ohne Rotation (reproduziert die Arbeit)
    rot_only          Rotation ohne Quantisierung (MUSS die Referenz treffen)
    rot_quant         Rotation + Quantisierung (der eigentliche Test)

``rot_only`` ist die Selbstkontrolle: Weicht sie von der Referenz ab, ist die
Rotation falsch implementiert und alle weiteren Zahlen sind wertlos.

Usage:
    python experiment_rotation.py --model Qwen/Qwen2-7B --nbits 4
    python experiment_rotation.py --analyze-only      # nur Kurtosis-Effekt
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

OUT = BASE / "results/raw/rotation"

# Globaler Zustand fuer den Attention-Patch: die Query-Rotation muss genau dann
# aktiv sein, wenn auch der Cache rotiert.
_ROT = {"H": None, "active": False}


def rotate(t: torch.Tensor) -> torch.Tensor:
    """Rotation in float32, Ergebnis zurueck im Ursprungs-Dtype.

    In float16 akkumuliert die Multiplikation ueber 128 Dimensionen einen
    relativen Fehler von ~5e-4. Bei Modellen mit Massive Activations werden
    daraus absolute Logit-Fehler, die der Softmax exponentiell verstaerkt --
    die Rotationsinvarianz waere dann nicht mehr messbar erfuellt.
    """
    H = _ROT["H"]
    return torch.matmul(t.float(), H.float()).to(t.dtype)


def hadamard(n: int, device, dtype) -> torch.Tensor:
    """Normierte Hadamard-Matrix (Sylvester-Konstruktion), n muss 2^k sein."""
    if n & (n - 1):
        raise ValueError(f"head_dim {n} ist keine Zweierpotenz")
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / n ** 0.5).to(device=device, dtype=dtype)


def install_rotation(model):
    """Rotiert Queries und Keys gemeinsam, direkt nach RoPE.

    Beide muessen an derselben Stelle gedreht werden. Haengt man die
    Query-Rotation in die Attention-Funktion und die Key-Rotation in den Cache,
    laufen sie auseinander, sobald ein Forward-Pass ohne Cache stattfindet
    (``use_cache=False``): Dann treffen rotierte Queries auf unrotierte Keys und
    die Scores stimmen nicht mehr. ``apply_rotary_pos_emb`` liefert genau den
    Punkt, an dem beide Tensoren gemeinsam vorliegen.
    """
    import importlib

    mod = importlib.import_module(type(model).__module__)
    orig = getattr(mod, "apply_rotary_pos_emb")
    if getattr(orig, "_rot_patched", False):
        return

    def patched(q, k, cos, sin, *a, **kw):
        q, k = orig(q, k, cos, sin, *a, **kw)
        if _ROT["active"] and _ROT["H"] is not None:
            q, k = rotate(q), rotate(k)
        return q, k

    patched._rot_patched = True
    mod.apply_rotary_pos_emb = patched
    print(f"Rotation eingehaengt in {mod.__name__}.apply_rotary_pos_emb")


def cache_factory(config, *, quantize: bool, nbits, axis, group_size, residual):
    """Cache-Fabrik. Die Rotation sitzt in apply_rotary_pos_emb, nicht hier --
    die Keys kommen also bereits rotiert an und werden im rotierten Raum
    quantisiert."""
    from transformers.cache_utils import Cache, DynamicLayer, HQQQuantizedLayer
    n = config.get_text_config(decoder=True).num_hidden_layers

    def factory():
        if quantize:
            layers = [HQQQuantizedLayer(nbits, axis, axis, group_size, residual)
                      for _ in range(n)]
        else:
            layers = [DynamicLayer() for _ in range(n)]
        return Cache(layers=layers)

    return factory


def excess_kurtosis(t: torch.Tensor) -> float:
    x = t.detach().float().flatten()
    z = (x - x.mean()) / x.std()
    return float((z ** 4).mean() - 3.0)


def analyze_rotation(model, tokenizer, H, device="cuda"):
    """Kurtosis der Keys vor und nach der Rotation, je Layer."""
    from transformers.cache_utils import DynamicCache
    ids = tokenizer("\n\n".join(["The quick brown fox jumps over the lazy dog."] * 64),
                    return_tensors="pt").input_ids[:, :2048].to(device)
    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        model(ids, past_key_values=cache, use_cache=True)
    rows = []
    for i, layer in enumerate(cache.layers):
        k = layer.keys
        if k is None or k.numel() == 0:
            continue
        kr = rotate(k)
        rows.append((i, excess_kurtosis(k), excess_kurtosis(kr),
                     float(k.abs().max()), float(kr.abs().max())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-7B")
    ap.add_argument("--nbits", type=int, default=4)
    ap.add_argument("--axis", type=int, default=1)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--residual", type=int, default=128)
    ap.add_argument("--ppl-tokens", type=int, default=4096)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    model, tokenizer, info = load_model(args.model, attn_backend="sdpa", device="cuda")
    model.eval()

    text_cfg = model.config.get_text_config(decoder=True)
    head_dim = getattr(text_cfg, "head_dim", None) or (
        text_cfg.hidden_size // text_cfg.num_attention_heads)
    print(f"head_dim = {head_dim}")
    H = hadamard(head_dim, "cuda", torch.float16)
    _ROT["H"] = H
    install_rotation(model)

    print("\nKurtosis-Effekt der Rotation (Key-Tensoren, 2048 Tokens):")
    rows = analyze_rotation(model, tokenizer, H)
    import statistics as st
    before = [r[1] for r in rows]
    after = [r[2] for r in rows]
    amax_b = [r[3] for r in rows]
    amax_a = [r[4] for r in rows]
    for i, b, a, mb, ma in rows[:6]:
        print(f"   Layer {i:>2}: kappa {b:9.2f} -> {a:7.2f} | "
              f"|max| {mb:8.1f} -> {ma:7.1f}")
    print(f"   ... ({len(rows)} Layer)")
    print(f"   Kurtosis  Mittel: {st.mean(before):8.2f}  ->  {st.mean(after):7.2f}")
    print(f"   Kurtosis  Max   : {max(before):8.2f}  ->  {max(after):7.2f}")
    print(f"   |max|     Mittel: {st.mean(amax_b):8.1f}  ->  {st.mean(amax_a):7.1f}")
    print(f"   |max|     Max   : {max(amax_b):8.1f}  ->  {max(amax_a):7.1f}")

    if args.analyze_only:
        return 0

    conds = {
        "fp16":       dict(rotate=False, quantize=False),
        "quant_only": dict(rotate=False, quantize=True),
        "rot_only":   dict(rotate=True,  quantize=False),
        "rot_quant":  dict(rotate=True,  quantize=True),
    }

    results = {}
    ppl_ref = None
    for name, cfg in conds.items():
        _ROT["active"] = cfg["rotate"]
        t0 = time.time()
        print(f"\nPPL [{name}] (rotate={cfg['rotate']}, quantize={cfg['quantize']}) ...")
        fac = None if not cfg["quantize"] else cache_factory(
            model.config, quantize=True, nbits=args.nbits, axis=args.axis,
            group_size=args.group_size, residual=args.residual)
        ppl = compute_perplexity(model, tokenizer, dataset="wikitext2",
                                 max_tokens=args.ppl_tokens, device="cuda",
                                 cache_factory=fac)
        if ppl_ref is None:
            ppl_ref = ppl
        results[name] = {"ppl": ppl, "delta": ppl - ppl_ref}
        print(f"  PPL = {ppl:.4f}   Delta = {ppl - ppl_ref:+.4f}   [{time.time()-t0:.0f}s]")
    _ROT["active"] = False

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = OUT / f"rotation_{args.model.split('/')[-1]}_int{args.nbits}_{stamp}.json"
    json.dump({"model": args.model, "nbits": args.nbits, "axis": args.axis,
               "q_group_size": args.group_size, "residual_length": args.residual,
               "head_dim": head_dim, "ppl_tokens": args.ppl_tokens,
               "kurtosis_before_mean": st.mean(before),
               "kurtosis_after_mean": st.mean(after),
               "kurtosis_per_layer": [{"layer": i, "kurt_before": b, "kurt_after": a,
                                       "absmax_before": mb, "absmax_after": ma}
                                      for i, b, a, mb, ma in rows],
               "results": results}, open(path, "w"), indent=2)

    print("\n" + "=" * 58)
    print(f"{'Bedingung':<14}{'PPL':>12}{'Delta':>12}")
    for name in conds:
        r = results[name]
        print(f"{name:<14}{r['ppl']:>12.4f}{r['delta']:>+12.4f}")
    print("=" * 58)
    drift = abs(results["rot_only"]["delta"])
    print(f"Selbstkontrolle rot_only: |Delta| = {drift:.4f} "
          f"({'OK' if drift < 0.05 else 'VERDAECHTIG - Rotation pruefen'})")
    print(f"Gespeichert: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
