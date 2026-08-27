#!/usr/bin/env python3
"""Kostet ``num_ctx`` Geschwindigkeit, oder nur Speicher?

Die Kontext-Kurve wurde bei *gefuelltem* Kontext gemessen: bei ``num_ctx``
196608 lagen auch 196k Token im Cache. Die Attention-Kosten je Token haengen
aber an der Zahl der tatsaechlich vorhandenen KV-Eintraege, nicht an der Groesse
des allokierten Puffers.

Falls das zutrifft, kostet ein gross konfiguriertes Fenster nur VRAM: eine
kurze Session liefe mit der Geschwindigkeit ihrer tatsaechlichen Laenge, und
der Durchsatz saenke erst, waehrend die Session waechst. Dann gaebe es kaum
einen Grund, ``num_ctx`` klein zu halten.

Falls nicht -- falls schon die Allokation drueckt -- ist ein kleines Fenster
im Alltag klar besser.

Der Test haelt die Fuellmenge fest und variiert nur ``num_ctx``:

    python test_allocation.py --host http://127.0.0.1:11435
"""

import argparse
import json
import statistics
import subprocess
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


def api_post(host, path, payload, timeout=1800.0):
    req = urllib.request.Request(
        f"{host.rstrip('/')}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def generate(host, model, prompt, num_ctx, num_predict, seed):
    return api_post(host, "/api/generate", {
        "model": model, "prompt": prompt, "stream": False, "think": False,
        "keep_alive": "10m",
        "options": {"num_ctx": num_ctx, "num_predict": num_predict,
                    "temperature": 0.0, "top_k": 1, "seed": seed},
    })


def unload(host, model):
    try:
        api_post(host, "/api/generate", {"model": model, "keep_alive": 0}, 120)
    except Exception:
        pass
    time.sleep(5)


def vram():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--id=0", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return 0.0


def build(fill_tokens, cpt, prefix):
    chars = int(max(fill_tokens - 64, 1) * cpt)
    reps = max(int(chars / len(FILLER)) + 1, 1)
    body = (FILLER * reps)[:chars]
    return prefix + body[len(prefix):] if len(body) > len(prefix) else prefix + body


def main():
    p = argparse.ArgumentParser(description="Allokation gegen Fuellstand")
    p.add_argument("--host", default="http://127.0.0.1:11435")
    p.add_argument("--model", default="qwen3.8:27b")
    p.add_argument("--fills", type=int, nargs="+", default=[8192, 32768])
    p.add_argument("--num-ctxs", type=int, nargs="+",
                   default=[8192, 32768, 131072, 262144])
    p.add_argument("--decode-tokens", type=int, default=512)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    if ":11434" in args.host:
        print("ABBRUCH: 11434 ist die Produktionsinstanz.", file=sys.stderr)
        return 2

    print("Kalibriere ...", flush=True)
    probe = FILLER * 40
    r = generate(args.host, args.model, probe, 8192, 1, args.seed)
    cpt = len(probe) / r["prompt_eval_count"]
    print(f"  {cpt:.3f} Zeichen/Token\n")

    print(f"{'Fuellung':>9} {'num_ctx':>9} {'Prompt-Tok':>11} "
          f"{'Decode':>10} {'VRAM':>10}")
    print("-" * 55)

    rows = []
    for fill in args.fills:
        for nctx in args.num_ctxs:
            if nctx < fill:
                continue
            # Neu laden: num_ctx aendert die Allokation, also muss das Modell
            # zwischen den Zellen raus.
            unload(args.host, args.model)
            base = vram()

            generate(args.host, args.model, build(fill, cpt, "[Warmup] "),
                     nctx, 8, args.seed)
            loaded = vram()

            tps, ptoks = [], 0
            for i in range(args.runs):
                # Eindeutiger Praefix gegen den Prompt-Cache.
                resp = generate(args.host, args.model,
                                build(fill, cpt, f"[Lauf {i}] "),
                                nctx, args.decode_tokens, args.seed + i)
                ev_ns, ev_ct = resp.get("eval_duration"), resp.get("eval_count")
                ptoks = resp.get("prompt_eval_count") or ptoks
                if ev_ns and ev_ct:
                    tps.append(ev_ct / (ev_ns / NS_PER_S))

            dec = statistics.median(tps) if tps else None
            rows.append({"fill": fill, "num_ctx": nctx, "prompt_tokens": ptoks,
                         "decode_tps": round(dec, 2) if dec else None,
                         "vram_loaded_mib": round(loaded, 1),
                         "vram_delta_mib": round(loaded - base, 1),
                         "decode_tps_runs": [round(v, 2) for v in tps]})
            print(f"{fill:>9} {nctx:>9} {ptoks:>11} "
                  f"{dec:>8.1f} t/s {loaded:>7.0f} MiB", flush=True)

    unload(args.host, args.model)

    print("\n" + "=" * 55)
    print("Auswertung: Decode bei gleicher Fuellung, verschiedenem num_ctx")
    print("=" * 55)
    for fill in args.fills:
        sub = [r for r in rows if r["fill"] == fill and r["decode_tps"]]
        if len(sub) < 2:
            continue
        vals = [r["decode_tps"] for r in sub]
        spread = (max(vals) - min(vals)) / min(vals) * 100
        print(f"\n  Fuellung {fill}: {vals}")
        print(f"  Spanne ueber num_ctx {sub[0]['num_ctx']}..{sub[-1]['num_ctx']}: "
              f"{spread:.1f}%")
        if spread < 8:
            print("  -> flach. Die Allokation kostet Speicher, aber keine "
                  "Geschwindigkeit.")
        else:
            print("  -> nicht flach. Schon das grosse Fenster drueckt den "
                  "Durchsatz, unabhaengig vom Fuellstand.")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump({"chars_per_token": cpt, "rows": rows}, fh,
                      indent=2, ensure_ascii=False)
        print(f"\nGeschrieben: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
