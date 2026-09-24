#!/usr/bin/env python3
"""Wertet die Messlaeufe aus und bestimmt die Pareto-Front.

Die Zielfrage ist ein Arbeitspunkt, nicht ein Datensatz: Welche Kombination aus
Gewichts- und KV-Quantisierung liefert bei welcher Kontextlaenge das beste
Verhaeltnis aus Qualitaet, Durchsatz und Speicher?

Bewertet wird deshalb entlang von vier Groessen:

- ``vram``      Gesamtbelegung aus den llama.cpp-Puffern (nicht nvidia-smi)
- ``decode``    Decode-Durchsatz, weil Kontextlaenge auch Geschwindigkeit kostet
- ``quality``   zusammengefasst aus Needle, Multi-Needle und Verbatim
- ``context``   die erreichte Kontextlaenge

Ein Punkt liegt auf der Pareto-Front, wenn kein anderer ihn in allen vier
Groessen zugleich erreicht oder uebertrifft.

Die Greedy-Divergenz wird gegen die f16-KV-Referenz derselben Gewichte
gerechnet: gleicher Prompt, gleiche Gewichte, greedy dekodiert -- jede
Abweichung geht damit auf die KV-Quantisierung zurueck.

    python aggregate_ollama.py                     # Tabellen auf stdout
    python aggregate_ollama.py --csv out.csv       # zusaetzlich als CSV
    python aggregate_ollama.py --markdown out.md
"""

import argparse
import difflib
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional


def load_runs(raw_dir: str) -> List[dict]:
    runs = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.json"))):
        try:
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception as exc:
            print(f"  uebersprungen (nicht lesbar): {os.path.basename(path)} -- {exc}",
                  file=sys.stderr)
            continue
        if d.get("runtime", {}).get("engine") != "ollama/llama.cpp":
            continue  # nur Laeufe dieses Messpfads
        d["_path"] = path
        runs.append(d)
    return runs


def quality_score(m: dict) -> Optional[float]:
    """Fasst die drei Sonden zu einer Zahl in [0, 1] zusammen.

    Gleichgewichtet, weil sie unterschiedliche Schadensbilder abdecken: Needle
    den Totalausfall, Multi-Needle die Teilverluste, Verbatim die schleichende
    Verfaelschung. Die Einzelwerte bleiben in der Tabelle sichtbar, damit die
    Zusammenfassung nichts verdeckt.
    """
    parts = []
    n = m.get("needle") or {}
    if n.get("success_rate") is not None:
        parts.append(n["success_rate"])
    q = m.get("quality") or {}
    mn = q.get("multi_needle") or {}
    if mn.get("hit_rate") is not None:
        parts.append(mn["hit_rate"])
    vb = q.get("verbatim") or {}
    if vb.get("similarity") is not None:
        parts.append(vb["similarity"])
    return round(sum(parts) / len(parts), 4) if parts else None


def greedy_divergence(cell: dict, reference: Optional[dict]) -> Optional[dict]:
    """Vergleicht die Greedy-Ausgaben gegen die f16-Referenz derselben Gewichte."""
    if reference is None:
        return None
    a = ((cell.get("quality") or {}).get("greedy_samples")) or []
    b = ((reference.get("quality") or {}).get("greedy_samples")) or []
    if not a or not b:
        return None

    sims, identical = [], 0
    for sa in a:
        sb = next((x for x in b if x.get("task_index") == sa.get("task_index")), None)
        if sb is None:
            continue
        ta, tb = sa.get("output", ""), sb.get("output", "")
        sims.append(difflib.SequenceMatcher(None, tb, ta).ratio())
        identical += int(ta == tb)
    if not sims:
        return None
    return {
        "tasks": len(sims),
        "identical": identical,
        "mean_similarity": round(sum(sims) / len(sims), 4),
        "min_similarity": round(min(sims), 4),
    }


def build_rows(runs: List[dict]) -> List[dict]:
    # Bei mehrfach gemessener Zelle gewinnt der juengste Lauf. Waehrend der
    # Entwicklung der Harness sind fehlerhafte Zellen entstanden (Prompt-Cache
    # im Prefill, kontaminiertes Log-Fenster im VRAM); ohne diese Regel stuenden
    # sie gleichberechtigt neben den korrigierten Werten.
    runs = sorted(runs, key=lambda r: r.get("timestamp", ""))
    newest: Dict[tuple, tuple] = {}
    for r in runs:
        kv = r.get("kv_quant", {}).get("kv_cache_type")
        for m in r.get("measurements", []):
            newest[(r["model"], kv, m.get("context_len"))] = (r, m)

    # Zellen indizieren, damit die f16-Referenz je (Modell, Kontext) auffindbar ist.
    by_key: Dict[tuple, dict] = {}
    for r in runs:
        kv = r.get("kv_quant", {}).get("kv_cache_type")
        for m in r.get("measurements", []):
            if "error" in m:
                continue
            by_key[(r["model"], kv, m["context_len"])] = m

    rows = []
    for (model, kv, ctx), (r, m) in sorted(newest.items(), key=lambda kv_: str(kv_[0])):
        # Library- und UD-Modell melden beide "Q4_K_M"; ohne Herkunft waeren
        # ihre Zeilen nicht auseinanderzuhalten.
        qlevel = (r.get("model_config") or {}).get("quantization_level") or "?"
        origin = "UD" if model.startswith("qwen3.8-ud") else "lib"
        weight = f"{qlevel} ({origin})"
        if True:
            if "error" in m:
                rows.append({
                    "model": model, "weight": weight, "kv": kv, "ctx": ctx,
                    "status": "FEHLER", "error": m.get("error", "")[:60],
                })
                continue
            b = m.get("buffers") or {}
            ref = by_key.get((model, "f16", ctx)) if kv != "f16" else None
            rows.append({
                "model": model,
                "weight": weight,
                "kv": kv,
                "ctx": ctx,
                "status": "ok",
                "vram_mib": b.get("total_gpu_mib"),
                "kv_mib": b.get("kv_mib"),
                "weights_mib": b.get("model_gpu_mib"),
                "prefill_tps": m.get("prefill_tokens_per_sec"),
                "decode_tps": m.get("decode_tokens_per_sec"),
                "needle": (m.get("needle") or {}).get("success_rate"),
                "multi_needle": ((m.get("quality") or {}).get("multi_needle") or {}).get("hit_rate"),
                "verbatim": ((m.get("quality") or {}).get("verbatim") or {}).get("similarity"),
                "quality": quality_score(m),
                "divergence": greedy_divergence(m, ref),
            })
    return rows


def pareto_front(rows: List[dict]) -> List[dict]:
    """Nicht-dominierte Punkte ueber Qualitaet, Decode, Kontext und VRAM."""
    ok = [r for r in rows
          if r["status"] == "ok" and None not in
          (r.get("quality"), r.get("decode_tps"), r.get("vram_mib"))]
    front = []
    for a in ok:
        dominated = False
        for b in ok:
            if a is b:
                continue
            # b dominiert a, wenn b nirgends schlechter und irgendwo besser ist.
            not_worse = (
                b["quality"] >= a["quality"]
                and b["decode_tps"] >= a["decode_tps"]
                and b["ctx"] >= a["ctx"]
                and b["vram_mib"] <= a["vram_mib"]
            )
            strictly_better = (
                b["quality"] > a["quality"]
                or b["decode_tps"] > a["decode_tps"]
                or b["ctx"] > a["ctx"]
                or b["vram_mib"] < a["vram_mib"]
            )
            if not_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(a)
    return sorted(front, key=lambda r: (-r["ctx"], -r["quality"]))


def fmt(v, spec="{:.0f}", dash="--"):
    return dash if v is None else spec.format(v)


def render_table(rows: List[dict], front_keys: set) -> str:
    hdr = ("| P | Gewichte | KV | Kontext | VRAM | davon KV | Prefill | Decode | "
           "Needle | Multi | Verbatim | Qualitaet | Divergenz |")
    sep = "|" + "---|" * 13
    out = [hdr, sep]
    for r in sorted(rows, key=lambda x: (x["weight"], x["kv"] or "", x["ctx"] or 0)):
        if r["status"] != "ok":
            out.append(f"| | {r['weight']} | {r['kv']} | {r['ctx']} | "
                       f"**{r['status']}** | {r.get('error','')} | | | | | | | |")
            continue
        key = (r["model"], r["kv"], r["ctx"])
        div = r.get("divergence")
        div_s = "Referenz" if r["kv"] == "f16" else (
            f"{div['mean_similarity']:.3f}" if div else "--")
        out.append(
            f"| {'**P**' if key in front_keys else ''} "
            f"| {r['weight']} | {r['kv']} | {r['ctx']} "
            f"| {fmt(r['vram_mib'])} MiB | {fmt(r['kv_mib'])} MiB "
            f"| {fmt(r['prefill_tps'])} | {fmt(r['decode_tps'], '{:.1f}')} "
            f"| {fmt(r['needle'], '{:.2f}')} | {fmt(r['multi_needle'], '{:.2f}')} "
            f"| {fmt(r['verbatim'], '{:.3f}')} | {fmt(r['quality'], '{:.3f}')} "
            f"| {div_s} |"
        )
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description="Auswertung der Ollama-Messlaeufe")
    p.add_argument("--raw-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..",
        "results", "ollama_probe", "raw"))
    p.add_argument("--csv", default=None)
    p.add_argument("--markdown", default=None)
    args = p.parse_args()

    runs = load_runs(args.raw_dir)
    if not runs:
        print(f"Keine Messlaeufe in {args.raw_dir}", file=sys.stderr)
        return 1

    rows = build_rows(runs)
    front = pareto_front(rows)
    front_keys = {(r["model"], r["kv"], r["ctx"]) for r in front}

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_err = len(rows) - n_ok
    header = (f"# Messergebnisse: Arbeitspunkt qwen3.8-27b\n\n"
              f"{len(runs)} Laeufe, {n_ok} Zellen ausgewertet, {n_err} Fehler "
              f"(OOM zaehlt als Ergebnis: markiert die Budgetgrenze).\n\n"
              f"**P** markiert die Pareto-Front ueber Qualitaet, Decode-Durchsatz, "
              f"Kontextlaenge und VRAM.\n\n")

    table = render_table(rows, front_keys)

    lines = [header, table, "\n## Pareto-Front\n"]
    if front:
        for r in front:
            lines.append(
                f"- **{r['weight']} + KV {r['kv']} @ {r['ctx']}** -- "
                f"Qualitaet {r['quality']:.3f}, {r['decode_tps']:.1f} tok/s, "
                f"{r['vram_mib']:.0f} MiB")
    else:
        lines.append("- (keine vollstaendigen Zellen)")

    text = "\n".join(lines)
    print(text)

    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"\nGeschrieben: {args.markdown}", file=sys.stderr)

    if args.csv:
        import csv
        cols = ["model", "weight", "kv", "ctx", "status", "vram_mib", "kv_mib",
                "weights_mib", "prefill_tps", "decode_tps", "needle",
                "multi_needle", "verbatim", "quality"]
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"Geschrieben: {args.csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
