#!/usr/bin/env python3
"""Welcher Anteil des Quantisierungsschadens entfaellt auf Keys, welcher auf Values?

Alle Messungen der Arbeit quantisieren Keys und Values gemeinsam. Die Analyse
argumentiert aber ausschliesslich ueber die *Key*-Verteilungen: Deren Kurtosis
sagt den Schaden vorher, und der Rotationstest zeigt, dass die Quantisierungs-
achse der Keys der eigentliche Hebel ist. Ob die Values ueberhaupt nennenswert
beitragen, ist damit offen -- und steht in Limitationen und Ausblick als offene
Frage.

Dieser Test trennt beide Seiten:

    fp16          Referenz
    both_stock    Keys und Values quantisiert, unveraenderte HQQ-Klasse
                  (muss den Wert aus der Arbeit reproduzieren)
    both_split    dasselbe, aber ueber die hier nachgebaute Layer-Klasse
                  (muss both_stock treffen)
    keys_only     nur Keys quantisiert, Values bleiben FP16
    values_only   nur Values quantisiert, Keys bleiben FP16

``both_split`` ist die Selbstkontrolle. Die Klasse ``SplitHQQLayer`` bildet die
Update-Logik von ``QuantizedLayer`` nach, weil das Original beide Seiten fest
verkoppelt. Weicht ``both_split`` von ``both_stock`` ab, ist die Nachbildung
falsch und die Split-Zahlen sind wertlos.

Zur Einordnung der Ersparnis: Keys und Values haben in jedem Layer dieselbe
Form. Wer nur eine Seite quantisiert, erhaelt daher exakt die Haelfte der
Cache-Reduktion.

Usage:
    python experiment_kv_split.py --model Qwen/Qwen3-8B --nbits 2
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

OUT = BASE / "results/raw/kv_split"


def _empty_like(t: torch.Tensor) -> torch.Tensor:
    return torch.tensor([], dtype=t.dtype, device=t.device)


def make_split_layer_cls():
    """Erzeugt die Layer-Klasse erst zur Laufzeit (Import braucht transformers)."""
    from transformers.cache_utils import HQQQuantizedLayer

    class SplitHQQLayer(HQQQuantizedLayer):
        """HQQ-Cache-Layer, der Keys und Values unabhaengig quantisieren kann.

        Nachbau von ``QuantizedLayer.update``. Der Unterschied liegt allein
        darin, dass die Entscheidung ueber Quantisierung und Residual-Puffer
        pro Seite faellt statt gemeinsam. Fuer eine nicht quantisierte Seite
        haelt ``self.keys`` bzw. ``self.values`` den vollstaendigen FP16-Cache,
        genau wie in ``DynamicLayer``.
        """

        def __init__(self, *args, quantize_keys=True, quantize_values=True, **kwargs):
            super().__init__(*args, **kwargs)
            self.quantize_keys = quantize_keys
            self.quantize_values = quantize_values

        def update(self, key_states, value_states, *args, **kwargs):
            self.cumulative_length += key_states.shape[-2]

            if not self.is_initialized:
                self.lazy_initialization(key_states, value_states)
                if self.quantize_keys:
                    self._quantized_keys = self._quantize(
                        key_states.contiguous(), axis=self.axis_key)
                else:
                    self.keys = key_states
                if self.quantize_values:
                    self._quantized_values = self._quantize(
                        value_states.contiguous(), axis=self.axis_value)
                else:
                    self.values = value_states
                return key_states, value_states

            if self.quantize_keys:
                keys_to_return = torch.cat(
                    [self._dequantize(self._quantized_keys), self.keys, key_states], dim=-2)
            else:
                keys_to_return = torch.cat([self.keys, key_states], dim=-2)

            if self.quantize_values:
                values_to_return = torch.cat(
                    [self._dequantize(self._quantized_values), self.values, value_states], dim=-2)
            else:
                values_to_return = torch.cat([self.values, value_states], dim=-2)

            # Residual-Puffer je Seite fuehren. Bei beidseitiger Quantisierung
            # sind beide Puffer formgleich, die Bedingung faellt dann mit der
            # des Originals zusammen.
            if self.quantize_keys:
                if self.keys.dim() == 4 and self.keys.shape[-2] + 1 >= self.residual_length:
                    self._quantized_keys = self._quantize(
                        keys_to_return.contiguous(), axis=self.axis_key)
                    self.keys = _empty_like(key_states)
                else:
                    self.keys = torch.cat([self.keys, key_states], dim=-2)
            else:
                self.keys = keys_to_return

            if self.quantize_values:
                if self.values.dim() == 4 and self.values.shape[-2] + 1 >= self.residual_length:
                    self._quantized_values = self._quantize(
                        values_to_return.contiguous(), axis=self.axis_value)
                    self.values = _empty_like(value_states)
                else:
                    self.values = torch.cat([self.values, value_states], dim=-2)
            else:
                self.values = values_to_return

            return keys_to_return, values_to_return

    return SplitHQQLayer


def cache_factory(config, mode, *, nbits, axis, group_size, residual):
    """Cache-Fabrik fuer eine der fuenf Bedingungen."""
    from transformers.cache_utils import Cache, DynamicLayer, HQQQuantizedLayer

    n = config.get_text_config(decoder=True).num_hidden_layers

    if mode == "fp16":
        return None

    if mode == "both_stock":
        def factory():
            return Cache(layers=[
                HQQQuantizedLayer(nbits, axis, axis, group_size, residual)
                for _ in range(n)])
        return factory

    Split = make_split_layer_cls()
    q_keys = mode in ("both_split", "keys_only")
    q_values = mode in ("both_split", "values_only")

    def factory():
        return Cache(layers=[
            Split(nbits, axis, axis, group_size, residual,
                  quantize_keys=q_keys, quantize_values=q_values)
            for _ in range(n)])

    return factory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--nbits", type=int, default=2)
    ap.add_argument("--axis", type=int, default=1)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--residual", type=int, default=128)
    ap.add_argument("--ppl-tokens", type=int, default=4096)
    args = ap.parse_args()

    model, tokenizer, info = load_model(args.model, attn_backend="sdpa", device="cuda")
    model.eval()

    modes = ["fp16", "both_stock", "both_split", "keys_only", "values_only"]
    results = {}
    ppl_ref = None

    for mode in modes:
        fac = cache_factory(model.config, mode, nbits=args.nbits, axis=args.axis,
                            group_size=args.group_size, residual=args.residual)
        t0 = time.time()
        print(f"\nPPL [{mode}] ...", flush=True)
        ppl = compute_perplexity(model, tokenizer, dataset="wikitext2",
                                 max_tokens=args.ppl_tokens, device="cuda",
                                 cache_factory=fac)
        if ppl_ref is None:
            ppl_ref = ppl
        results[mode] = {"ppl": ppl, "delta": ppl - ppl_ref}
        print(f"  PPL = {ppl:.4f}   Delta = {ppl - ppl_ref:+.4f}   [{time.time()-t0:.0f}s]",
              flush=True)
        torch.cuda.empty_cache()

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = OUT / f"kv_split_{args.model.split('/')[-1]}_int{args.nbits}_{stamp}.json"
    json.dump({"model": args.model, "nbits": args.nbits, "axis": args.axis,
               "q_group_size": args.group_size, "residual_length": args.residual,
               "ppl_tokens": args.ppl_tokens, "results": results},
              open(path, "w"), indent=2)

    print("\n" + "=" * 58)
    print(f"{'Bedingung':<14}{'PPL':>12}{'Delta':>12}")
    for m in modes:
        print(f"{m:<14}{results[m]['ppl']:>12.4f}{results[m]['delta']:>+12.4f}")
    print("=" * 58)

    drift = abs(results["both_split"]["delta"] - results["both_stock"]["delta"])
    ok = drift < 0.05
    print(f"Selbstkontrolle both_split vs both_stock: Abweichung {drift:.4f} "
          f"({'OK' if ok else 'VERDAECHTIG - Nachbildung pruefen'})")

    both = results["both_stock"]["delta"]
    if abs(both) > 1e-9:
        print(f"\nAnteil am Gesamtschaden (Delta_both = {both:.4f}):")
        for m in ("keys_only", "values_only"):
            print(f"  {m:<12} {results[m]['delta']:>10.4f}  "
                  f"= {100 * results[m]['delta'] / both:5.1f} %")
    print(f"Gespeichert: {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
