#!/usr/bin/env python3
"""Kurzmessung des Decode-Durchsatzes bei gefuelltem Kontext.

Bewusst abgespeckt gegenueber ``bench_ollama.py``: keine Qualitaetssonden, kein
Needle, und der Prompt bleibt zwischen den Laeufen gleich, sodass der Prefill den
Cache trifft. Fuer den Decode aendert das nichts -- er wird ueber
``eval_count``/``eval_duration`` gemessen und laeuft in jedem Lauf ueber den
vollen, gefuellten Kontext. Gespart wird nur die Wartezeit fuer wiederholte
Prefills, die bei 131k rund 83 Sekunden je Lauf kosten.

Gedacht fuer Vergleiche, bei denen sich nur eine Variable aendert (Version,
Flag, Treiber) und die volle Matrix nicht noetig ist.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.request

FILLER = (
    "Die Verwaltung des Gebaeudes plant die Wartung der Lueftungsanlage. "
    "Mitarbeiterinnen und Mitarbeiter werden gebeten, die Hinweise am "
    "schwarzen Brett zu beachten. Der Empfang bleibt waehrend der "
    "Umbauphase durchgehend besetzt. "
)
NS_PER_S = 1_000_000_000


def gen(host, model, prompt, num_ctx, num_predict, seed):
    payload = {
        "model": model, "prompt": prompt, "stream": False, "think": False,
        "keep_alive": "10m",
        "options": {"num_ctx": num_ctx, "num_predict": num_predict,
                    "temperature": 0.0, "top_k": 1, "seed": seed},
    }
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read().decode())


def main():
    p = argparse.ArgumentParser(description="Kurzmessung Decode-Durchsatz")
    p.add_argument("--host", default="http://127.0.0.1:11435")
    p.add_argument("--model", default="qwen3.8:27b")
    p.add_argument("--context", type=int, default=131072)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--decode-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if ":11434" in args.host:
        print("ABBRUCH: 11434 ist die Produktionsinstanz.", file=sys.stderr)
        return 2

    # Kalibrieren, damit der Kontext wirklich gefuellt wird.
    probe = FILLER * 40
    r = gen(args.host, args.model, probe, 8192, 1, args.seed)
    cpt = len(probe) / r["prompt_eval_count"]

    chars = int((args.context - 600) * cpt)
    body = (FILLER * (int(chars / len(FILLER)) + 1))[:chars]
    prompt = body + "\n\nFasse den obigen Text in drei Saetzen zusammen."

    t0 = time.perf_counter()
    warm = gen(args.host, args.model, prompt, args.context, 8, args.seed)
    load_s = time.perf_counter() - t0
    filled = warm.get("prompt_eval_count", 0)
    print(f"    Warmup: {filled} Token im Kontext, {load_s:.0f} s (inkl. Modell-Load)")

    tps = []
    for i in range(args.runs):
        resp = gen(args.host, args.model, prompt, args.context,
                   args.decode_tokens, args.seed + i)
        ev_ns, ev_ct = resp.get("eval_duration"), resp.get("eval_count")
        if ev_ns and ev_ct:
            tps.append(ev_ct / (ev_ns / NS_PER_S))

    if not tps:
        print("    keine verwertbaren Timings", file=sys.stderr)
        return 1

    med = statistics.median(tps)
    spread = (max(tps) - min(tps)) / min(tps) * 100
    print(f"    Decode: {med:.1f} tok/s   "
          f"(Einzelwerte {', '.join(f'{v:.1f}' for v in tps)}, Spanne {spread:.1f} %)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
