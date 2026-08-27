#!/usr/bin/env python3
"""Stellt Ollama/llama.cpp und vLLM nebeneinander.

Die beiden Messpfade schreiben dasselbe Schema, unterscheiden sich aber in zwei
Punkten, die beim Vergleich zu beachten sind:

- **Speicher.** llama.cpp meldet seine Puffer einzeln (Gewichte, KV, SSM-State,
  Compute); vLLM allokiert ueber ``gpu-memory-utilization`` vorab einen Pool und
  schluesselt nicht auf. Die VRAM-Spalte ist deshalb nur innerhalb eines
  Runtimes vergleichbar. Fuer den Vergleich zaehlt die Groesse der Gewichte,
  die aus dem Checkpoint bekannt ist.
- **Phasentrennung.** Ollama liefert ``prompt_eval_duration`` und
  ``eval_duration`` getrennt; bei vLLM wird die Zeit bis zum ersten Token als
  Prefill gewertet. In der TTFT steckt etwas Scheduling, was bei Prefills von
  einer bis vier Minuten aber nicht ins Gewicht faellt.

Direkt vergleichbar sind damit **Decode-Durchsatz und Qualitaet** -- und genau
daran haengt die Frage, ob ein Stack-Wechsel lohnt.

    python compare_runtimes.py
"""

import glob
import json
import os
import sys

# Gewichtsgroessen der Checkpoints, aus den HF-Dateilisten. Der VRAM-Wert der
# Messung taugt fuer den Runtime-Vergleich nicht, diese Zahl schon.
WEIGHTS_GIB = {
    "qwen3.8:27b": 15.33,
    "qwen3.8-udv:27b-q4_K_M": 15.33,
    "qwen3.8-ud:27b-q4_K_M": 15.33,
    "nvfp4": 21.81,
    "int4": 18.12,
}


def quality(m):
    parts = []
    n = m.get("needle") or {}
    if n.get("success_rate") is not None:
        parts.append(n["success_rate"])
    q = m.get("quality") or {}
    for key, field in (("multi_needle", "hit_rate"), ("verbatim", "similarity")):
        v = (q.get(key) or {}).get(field)
        if v is not None:
            parts.append(v)
    return round(sum(parts) / len(parts), 4) if parts else None


def load(base):
    rows = []
    for sub in ("raw", "raw_vllm"):
        for f in glob.glob(os.path.join(base, sub, "*.json")):
            try:
                d = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            eng = (d.get("runtime") or {}).get("engine", "?")
            if eng.startswith("vllm"):
                label = (d.get("runtime") or {}).get("checkpoint") or "vllm"
                kv = "fp8"
                weights = WEIGHTS_GIB.get(label)
            else:
                label = d.get("model", "?")
                kv = (d.get("kv_quant") or {}).get("kv_cache_type", "?")
                weights = WEIGHTS_GIB.get(label)
            for m in d.get("measurements", []):
                if "error" in m:
                    continue
                b = m.get("buffers") or {}
                rows.append({
                    "engine": "vLLM" if eng.startswith("vllm") else "Ollama",
                    "label": label, "kv": kv,
                    "ctx": m.get("context_len"),
                    "prefill": m.get("prefill_tokens_per_sec"),
                    "decode": m.get("decode_tokens_per_sec"),
                    "vram": b.get("total_gpu_mib") or m.get("vram_after_load_mib"),
                    "weights_gib": weights,
                    "quality": quality(m),
                    "decode_tokens": (d.get("config") or {}).get("decode_tokens"),
                })
    return rows


def main():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "results", "ollama_probe")
    rows = load(base)
    if not rows:
        print(f"Keine Daten unter {base}", file=sys.stderr)
        return 1

    # Nur das aktuelle Protokoll: 128-Token-Zellen sind wegen der schwankenden
    # MTP-Akzeptanz nicht mit den 512er-Zellen mischbar.
    rows = [r for r in rows if r.get("decode_tokens") == 512]
    if not rows:
        print("Keine Zellen mit 512 Decode-Token gefunden.", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: (r["ctx"] or 0, r["engine"], r["label"]))

    print("# Runtime-Vergleich (nur Zellen mit 512 Decode-Token)\n")
    print("| Kontext | Runtime | Checkpoint | KV | Gewichte | Decode | Prefill | Qualität |")
    print("|---:|---|---|---|---:|---:|---:|---:|")
    for r in rows:
        w = f"{r['weights_gib']:.2f} GiB" if r["weights_gib"] else "—"
        print(f"| {r['ctx']} | {r['engine']} | {r['label']} | `{r['kv']}` | {w} | "
              f"**{r['decode']:.1f}** | {r['prefill']:.0f} | "
              f"{r['quality']:.3f} |" if r["decode"] and r["quality"] is not None
              else f"| {r['ctx']} | {r['engine']} | {r['label']} | `{r['kv']}` | {w} | — | — | — |")

    # Direkter Gegenueberstellung je Kontextlaenge, sofern beide Runtimes messen.
    print("\n## Gegenüberstellung je Kontextlänge\n")
    for ctx in sorted({r["ctx"] for r in rows}):
        sub = [r for r in rows if r["ctx"] == ctx and r["decode"]]
        oll = [r for r in sub if r["engine"] == "Ollama"]
        vll = [r for r in sub if r["engine"] == "vLLM"]
        if not (oll and vll):
            continue
        best_o = max(oll, key=lambda r: r["decode"])
        print(f"**{ctx} Token** — Ollama bestes: {best_o['label']} `{best_o['kv']}` "
              f"mit {best_o['decode']:.1f} tok/s")
        for v in sorted(vll, key=lambda r: -r["decode"]):
            diff = (v["decode"] - best_o["decode"]) / best_o["decode"] * 100
            extra = (v["weights_gib"] or 0) - (best_o["weights_gib"] or 0)
            print(f"  - vLLM {v['label']}: {v['decode']:.1f} tok/s "
                  f"({diff:+.1f} %), {extra:+.2f} GiB Gewichte, "
                  f"Qualität {v['quality']:.3f}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
