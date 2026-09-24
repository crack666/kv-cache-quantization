#!/usr/bin/env python3
"""Misst die beiden Budgetposten, die neben dem Chat-Modell auf der Karte liegen.

Bisher standen dafuer nur Schaetzungen: der Embedder mit ~3,8 GB aus
``compose.yaml`` und der Windows-Desktop mit einer im Verlauf beobachteten
Spanne von 1,1--4,4 GB. An beiden Zahlen haengt, welche Kontextlaenge produktiv
ueberhaupt passt -- sie sollten gemessen sein.

    python measure_budget.py --host http://127.0.0.1:11435
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request


def vram_used_mib(gpu_index: int = 0) -> float:
    out = subprocess.run(
        ["nvidia-smi", f"--id={gpu_index}",
         "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    )
    return float(out.stdout.strip().splitlines()[0])


def gpu_processes() -> list:
    """Was ausser dem Modell noch auf der Karte liegt."""
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory,name",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10,
    )
    return [l.strip() for l in out.stdout.strip().splitlines() if l.strip()]


def api_post(host: str, path: str, payload: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(
        f"{host.rstrip('/')}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main() -> int:
    p = argparse.ArgumentParser(description="Embedder- und Desktop-Fussabdruck messen")
    p.add_argument("--host", default="http://127.0.0.1:11435")
    p.add_argument("--embed-model", default="qwen3-embedding:4b-ctx2k")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    if ":11434" in args.host:
        print("ABBRUCH: 11434 ist die Produktionsinstanz.", file=sys.stderr)
        return 2

    result = {}

    print("=" * 68)
    print("1. Desktop-Grundlast (nichts geladen)")
    print("=" * 68)
    baseline = vram_used_mib()
    result["desktop_baseline_mib"] = baseline
    result["gpu_processes"] = gpu_processes()
    print(f"  belegt: {baseline:.0f} MiB")
    for line in result["gpu_processes"]:
        print(f"    {line}")

    print()
    print("=" * 68)
    print(f"2. Embedder-Fussabdruck ({args.embed_model})")
    print("=" * 68)
    try:
        # keep_alive gross setzen: die Frage ist auch, ob er resident bleibt.
        api_post(args.host, "/api/embed",
                 {"model": args.embed_model, "input": "Testsatz zur Messung.",
                  "keep_alive": "10m"})
        time.sleep(3)
        loaded = vram_used_mib()
        delta = loaded - baseline
        result["embedder_loaded_mib"] = loaded
        result["embedder_delta_mib"] = delta
        print(f"  belegt nach Laden: {loaded:.0f} MiB")
        print(f"  Fussabdruck:       {delta:.0f} MiB  ({delta/1024:.2f} GiB)")

        ps = json.loads(urllib.request.urlopen(
            f"{args.host.rstrip('/')}/api/ps", timeout=30).read().decode())
        for m in ps.get("models", []):
            print(f"  /api/ps: {m.get('name')} -> "
                  f"{(m.get('size_vram') or 0)/2**20:.0f} MiB "
                  f"(unterschaetzt systematisch, siehe FINDINGS)")
    except Exception as exc:
        print(f"  FEHLER: {exc}", file=sys.stderr)
        result["embedder_error"] = str(exc)[:200]

    print()
    print("=" * 68)
    print("3. Budgetrechnung")
    print("=" * 68)
    total = 32607.0
    emb = result.get("embedder_delta_mib") or 0.0
    # Aus zwei Messpunkten (128k = 21522, 256k = 25243 MiB) linear gefittet.
    base0, per_token_kib = 17797.0, 29.1
    print(f"  Karte gesamt          {total:>8.0f} MiB")
    print(f"  Desktop (jetzt)       {baseline:>8.0f} MiB")
    print(f"  Embedder              {emb:>8.0f} MiB")
    free = total - baseline - emb
    print(f"  frei fuers Chat-Modell{free:>8.0f} MiB")
    if free > base0:
        max_ctx = (free - base0) * 1024 / per_token_kib
        print(f"\n  -> maximale Kontextlaenge bei q4_0: {max_ctx:,.0f} Token")
        for reserve in (1024, 2048):
            c = (free - reserve - base0) * 1024 / per_token_kib
            print(f"     mit {reserve/1024:.0f} GB Reserve:            {c:,.0f} Token")
    else:
        print("\n  -> Modell passt so nicht mehr.")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        print(f"\nGeschrieben: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
