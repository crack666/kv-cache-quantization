"""
pareto_analysis.py — Pareto-Front der Qualitäts-Speicher-Trade-offs.

Liest die Long-Context-Summaries (ctx = 32k) und bestimmt pro Modell die
Pareto-Front über den Konfigurationsraum {FP16, INT8, INT4, INT2, INT2-KIVI}
mit den Zielen:
  - KV-Cache-Größe (kv_mb, minimieren)
  - Qualitätsverlust (|Δ-PPL|, minimieren)

Ausgaben:
  1. pareto_front.pdf/.png  — Scatter + Front pro Modell (log-log)
  2. LaTeX-Tabelle "Arbeitspunkte je VRAM-Budget" auf stdout
     (bester Qualitätspunkt, dessen VRAM-Peak ins Budget passt)

Dominanz: Konfiguration A dominiert B, wenn A in beiden Zielen <= B ist
und in mindestens einem strikt besser.
"""

import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

BASE = Path(__file__).parent.parent
LONG_CTX = BASE / "results/raw/long_context_final"
OUT_DIR = BASE / "results/figures/thesis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

MODEL_LABELS = {
    "google/gemma-4-E4B":        "Gemma-4-E4B",
    "mistralai/Mistral-7B-v0.1": "Mistral-7B",
    "01-ai/Yi-1.5-9B":           "Yi-1.5-9B",
    "Qwen/Qwen3-8B":             "Qwen3-8B",
    "Qwen/Qwen2-7B":             "Qwen2-7B",
}
MODEL_COLORS = {
    "Gemma-4-E4B": "#2E7D32",
    "Mistral-7B":  "#1565C0",
    "Yi-1.5-9B":   "#6A1B9A",
    "Qwen3-8B":    "#E65100",
    "Qwen2-7B":    "#B71C1C",
}
MODEL_MARKERS = {
    "Gemma-4-E4B": "o",
    "Mistral-7B":  "s",
    "Yi-1.5-9B":   "D",
    "Qwen3-8B":    "^",
    "Qwen2-7B":    "v",
}
QUANT_LABELS = {"fp16": "FP16", "int8-hqq": "INT8", "int4-hqq": "INT4",
                "int2-hqq": "INT2", "int2-hqq-kivi": "INT2-KIVI",
                "int2-hqq(kivi)": "INT2-KIVI"}
QUANT_COLORS = {"fp16": "#4878CF", "int8-hqq": "#6ACC65", "int4-hqq": "#D65F5F",
                "int2-hqq": "#B47CC7", "int2-hqq(kivi)": "#C4AD66",
                "int2-hqq-kivi": "#C4AD66"}

DPPL_FLOOR = 1e-3      # log-Darstellung: |Δ|=0 (FP16-Baseline) wird geklemmt
QUALITY_CAP = 1.0      # praktische Front: |Δ-PPL| < 1.0 (Konsistenz mit Fig. Kurtosis)
BUDGETS_GB = [16, 24, 32]


def load_points():
    """Ein Punkt je (Modell, Quant-Konfiguration) bei maximaler Kontextlänge."""
    points = {}
    for f in sorted(glob.glob(str(LONG_CTX / "*_summary.json"))):
        d = json.load(open(f))
        label = MODEL_LABELS.get(d["model"], d["model"])
        combos = d["combinations"]
        max_ctx = max(c["ctx"] for c in combos)
        for c in combos:
            if c["ctx"] != max_ctx:
                continue
            q = c["kv_quant"]
            dppl = abs(c["ppl_delta"]) if c["ppl_delta"] is not None else 0.0
            points.setdefault(label, []).append({
                "quant": q,
                "kv_mb": c["kv_mb"],
                "dppl": dppl,
                "vram_mb": c["vram_peak_mb"],
                "decode_tok_s": c["decode_tok_s"],
                "ctx": c["ctx"],
            })
    return points


def pareto_front(pts):
    """Nicht-dominierte Punkte (kv_mb, dppl), beides minimieren."""
    front = []
    for a in pts:
        dominated = any(
            (b["kv_mb"] <= a["kv_mb"] and b["dppl"] <= a["dppl"]) and
            (b["kv_mb"] < a["kv_mb"] or b["dppl"] < a["dppl"])
            for b in pts
        )
        if not dominated:
            front.append(a)
    return sorted(front, key=lambda p: p["kv_mb"])


def plot(points):
    """Small Multiples: ein Panel pro Modell, Punkte nach Bitbreite gefärbt.

    Gefüllt = pareto-optimal (Front, graue Stufenlinie), offen = dominiert.
    Gemeinsame log-Achsen machen die Panels direkt vergleichbar.
    """
    order = ["Gemma-4-E4B", "Mistral-7B", "Yi-1.5-9B", "Qwen3-8B", "Qwen2-7B"]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.8), sharex=True, sharey=True)
    axes = axes.flatten()

    # gemeinsame Achsengrenzen
    all_x = [p["kv_mb"] for pts in points.values() for p in pts]
    all_y = [max(p["dppl"], DPPL_FLOOR) for pts in points.values() for p in pts]
    xlim = (min(all_x) * 0.6, max(all_x) * 1.8)
    ylim = (DPPL_FLOOR * 0.5, max(all_y) * 8)

    for ax, model in zip(axes, order):
        pts = points[model]
        front = pareto_front(pts)
        front_set = {(p["kv_mb"], p["quant"]) for p in front}

        fx = [p["kv_mb"] for p in front]
        fy = [max(p["dppl"], DPPL_FLOOR) for p in front]
        ax.plot(fx, fy, "--", color="gray", alpha=0.8, zorder=2, linewidth=1.2)

        for p in pts:
            x, y = p["kv_mb"], max(p["dppl"], DPPL_FLOOR)
            on_front = (p["kv_mb"], p["quant"]) in front_set
            color = QUANT_COLORS[p["quant"]]
            ax.scatter(x, y, s=75 if on_front else 45, marker="o", color=color,
                       facecolors=color if on_front else "none",
                       linewidths=1.4, zorder=3, alpha=0.95 if on_front else 0.6)

        ax.axhline(0.01, color="green", linestyle=":", alpha=0.5, linewidth=0.9)
        ax.axhline(QUALITY_CAP, color="red", linestyle=":", alpha=0.5, linewidth=0.9)
        ax.text(xlim[1] * 0.75, 0.0115, "verlustfrei", fontsize=7, color="green",
                ha="right", va="bottom")
        ax.text(xlim[1] * 0.75, 1.18, "kritisch", fontsize=7, color="red",
                ha="right", va="bottom")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        # Lesbare Ticks: echte MB-Werte statt 10er-Potenzen
        ax.xaxis.set_major_locator(mticker.FixedLocator([100, 300, 1000, 3000]))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ctx_note = f"{pts[0]['ctx'] // 1024}k"
        ax.set_title(f"{model} (ctx = {ctx_note})", fontsize=10.5,
                     color=MODEL_COLORS[model])

    # Jedes Panel bekommt eigene Tick-Labels auf beiden Achsen. Mit sharex/sharey
    # beschriftet Matplotlib sonst nur die unterste Zeile und die linke Spalte,
    # sodass vier der fuenf Panels ohne ablesbare Achse dastehen.
    for ax in axes[:5]:
        ax.tick_params(labelbottom=True, labelleft=True)

    # 6. Panel: Legende + Lesehilfe
    lax = axes[5]
    lax.axis("off")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=QUANT_COLORS[q],
                          markersize=8, label=QUANT_LABELS[q])
               for q in ["fp16", "int8-hqq", "int4-hqq", "int2-hqq", "int2-hqq(kivi)"]]
    handles += [
        plt.Line2D([], [], marker="o", linestyle="", color="gray",
                   markersize=8, label="gefüllt: pareto-optimal"),
        plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
                   color="gray", markersize=8, label="offen: dominiert"),
        plt.Line2D([], [], linestyle="--", color="gray", label="Pareto-Front"),
        plt.Line2D([], [], linestyle=":", color="green",
                   label="verlustfrei ($|\\Delta| = 0{,}01$)"),
        plt.Line2D([], [], linestyle=":", color="red",
                   label="kritisch ($|\\Delta| = 1{,}0$)"),
    ]
    lax.legend(handles=handles, loc="center", frameon=False, fontsize=9.5)

    fig.supxlabel("KV-Cache-Größe (MB, log)", fontsize=11)
    fig.supylabel("$|\\Delta$-PPL$|$ gegenüber FP16 (log)", fontsize=11)
    fig.suptitle("Pareto-Analyse: Qualitätsverlust vs. KV-Cache-Größe", fontsize=12.5)

    # h_pad hält die X-Beschriftung der oberen Reihe von den Titeln darunter frei,
    # w_pad schafft Platz fuer die nun in jeder Spalte gesetzten Y-Labels
    fig.tight_layout(h_pad=2.4, w_pad=1.8)
    for ext, kw in [("pdf", {}), ("png", {"dpi": 150})]:
        out = OUT_DIR / f"pareto_front.{ext}"
        fig.savefig(out, bbox_inches="tight", **kw)
        print(f"Saved: {out}")


def budget_table(points):
    """LaTeX-Zeilen: bester Arbeitspunkt (min |Δ-PPL|) je Modell und VRAM-Budget.

    Nutzbarkeits-Kriterium: Decode >= 1 tok/s. Schließt formal passende, aber
    praktisch unbenutzbare Konfigurationen aus (z.B. Mistral FP16 @32k:
    VRAM-Peak 30,9 GB passt in 32 GB, aber PCIe-Swap drückt auf 0,03 tok/s).
    """
    MIN_TOK_S = 1.0
    print("\n% Arbeitspunkte je VRAM-Budget (bester Qualitätspunkt im Budget,")
    print("% nutzbar = Decode >= 1 tok/s; ctx = max. gemessene Länge)")
    header = "\\textbf{Modell} & " + " & ".join(
        f"\\textbf{{{b}\\,GB}}" for b in BUDGETS_GB) + " \\\\"
    print(header + "\n\\midrule")
    for model, pts in sorted(points.items()):
        cells = []
        for budget in BUDGETS_GB:
            fitting = [p for p in pts
                       if p["vram_mb"] <= budget * 1024
                       and p["decode_tok_s"] >= MIN_TOK_S]
            if not fitting:
                cells.append("--")
                continue
            best = min(fitting, key=lambda p: (p["dppl"], p["kv_mb"]))
            marker = "" if best["dppl"] < QUALITY_CAP else "$^\\dagger$"
            cells.append(f"{QUANT_LABELS[best['quant']]}{marker}")
        print(f"{model} & " + " & ".join(cells) + " \\\\")
    print("% $^\\dagger$ = |Δ-PPL| >= 1.0 (kritisch), nur formal optimal\n")


def front_report(points):
    print("\nPareto-Fronten (formal) und praktische Front (|Δ-PPL| < 1.0):")
    for model, pts in sorted(points.items()):
        front = pareto_front(pts)
        practical = [p for p in front if p["dppl"] < QUALITY_CAP]
        fmt = lambda ps: ", ".join(
            f"{QUANT_LABELS[p['quant']]} ({p['kv_mb']:.0f} MB, |Δ|={p['dppl']:.4g})" for p in ps)
        print(f"  {model}:")
        print(f"    formal:    {fmt(front)}")
        print(f"    praktisch: {fmt(practical) if practical else '(leer)'}")


if __name__ == "__main__":
    # Windows-Konsole gibt sonst cp1252 aus und scheitert am Delta der Reports
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    pts = load_points()
    for m, p in pts.items():
        print(f"{m}: {len(p)} Konfigurationen @ctx={p[0]['ctx']}")
    plot(pts)
    front_report(pts)
    budget_table(pts)
