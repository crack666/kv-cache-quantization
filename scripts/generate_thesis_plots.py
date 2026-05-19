"""
generate_thesis_plots.py — Thesis-ready figures for KV-Cache quantization study.

Generates 4 plots:
  1. kurtosis_vs_ppl_delta.pdf  — Key-Kurtosis vs Δ-PPL scatter (INT2, all models)
  2. vram_vs_context.pdf        — VRAM peak vs context length per model × quant
  3. delta_ppl_heatmap.pdf      — Δ-PPL heatmap (models × quant levels)
  4. needle_comparison.pdf      — Needle-in-a-Haystack scores (models × quant levels)

Output: ../results/figures/thesis/
"""

import json
import glob
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
LONG_CTX = BASE / "results/raw/long_context"
KV_DIST  = BASE / "results/raw/kv_distributions_v2"
OUT_DIR  = BASE / "results/figures/thesis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name: str, idx: int):
    """Save fig as PDF (with fallback) and always also as PNG at 150 Dpi."""
    stem = Path(name).stem
    # PDF
    for suffix in [f"{stem}.pdf", f"{stem}_new.pdf"]:
        out = OUT_DIR / suffix
        try:
            fig.savefig(out, bbox_inches="tight")
            print(f"  [{idx}] Saved: {out}")
            break
        except PermissionError:
            continue
    else:
        print(f"  [{idx}] ERROR: could not save {name} (file locked)")
    # PNG (always writable — different extension)
    png_out = OUT_DIR / f"{stem}.png"
    fig.savefig(png_out, bbox_inches="tight", dpi=150)

# ── Style ─────────────────────────────────────────────────────────────────────
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
    "axes.spines.right":False,
})

MODEL_LABELS = {
    "google/gemma-4-E4B":         "Gemma-4-E4B",
    "mistralai/Mistral-7B-v0.1":  "Mistral-7B",
    "01-ai/Yi-1.5-9B":            "Yi-1.5-9B",
    "Qwen/Qwen3-8B":              "Qwen3-8B",
    "Qwen/Qwen2-7B":              "Qwen2-7B",
}

QUANT_ORDER   = ["fp16", "int8-hqq", "int4-hqq", "int2-hqq", "int2-hqq-kivi"]
QUANT_LABELS  = {"fp16": "FP16", "int8-hqq": "INT8", "int4-hqq": "INT4",
                 "int2-hqq": "INT2", "int2-hqq-kivi": "INT2-KIVI"}
QUANT_COLORS  = {"fp16": "#4878CF", "int8-hqq": "#6ACC65", "int4-hqq": "#D65F5F",
                 "int2-hqq": "#B47CC7", "int2-hqq-kivi": "#C4AD66"}
MODEL_COLORS  = {
    "Gemma-4-E4B": "#E63946",
    "Mistral-7B":  "#457B9D",
    "Yi-1.5-9B":   "#2A9D8F",
    "Qwen3-8B":    "#E9C46A",
    "Qwen2-7B":    "#F4A261",
}
MODEL_MARKERS = {
    "Gemma-4-E4B": "D",
    "Mistral-7B":  "o",
    "Yi-1.5-9B":   "s",
    "Qwen3-8B":    "^",
    "Qwen2-7B":    "v",
}

# ── Data loading ───────────────────────────────────────────────────────────────

def load_summaries():
    """Returns dict: model_label -> list of combo dicts."""
    result = {}
    for f in glob.glob(str(LONG_CTX / "*_summary.json")):
        d = json.load(open(f))
        label = MODEL_LABELS.get(d["model"], d["model"])
        result[label] = d["combinations"]
    return result


def load_kv_dists():
    """Returns dict: model_label -> summary dict."""
    result = {}
    for f in glob.glob(str(KV_DIST / "*.json")):
        d = json.load(open(f))
        label = MODEL_LABELS.get(d["model"], d["model"])
        result[label] = d["summary"]
    return result


# ── Plot 1: Kurtosis vs Δ-PPL scatter ─────────────────────────────────────────

def plot_kurtosis_vs_ppl(summaries, kv_dists):
    from scipy.stats import spearmanr

    fig, ax = plt.subplots(figsize=(9, 6))

    # ── Risk-zone shading ──────────────────────────────────────────────────────
    ax.axhspan(1e-4, 0.01, color="#2ca02c", alpha=0.08, zorder=0)   # green: verlustfrei
    ax.axhspan(0.01, 1.0,  color="#ff7f0e", alpha=0.08, zorder=0)   # yellow: akzeptabel
    ax.axhspan(1.0,  1e5,  color="#d62728", alpha=0.08, zorder=0)   # red: kritisch
    ax.axhline(0.01, color="#2ca02c", linestyle="--", linewidth=0.9,
               alpha=0.7, label="|Δ-PPL| = 0.01 (verlustfrei)", zorder=1)
    ax.axhline(1.0,  color="#d62728", linestyle="--", linewidth=0.9,
               alpha=0.7, label="|Δ-PPL| = 1.0 (kritisch)",     zorder=1)

    # ── Collect data for correlation and plotting ──────────────────────────────
    quant_styles = {
        "int2-hqq": dict(filled=True,  size=140, label_suffix=" INT2"),
        "int4-hqq": dict(filled=False, size=140, label_suffix=" INT4"),
    }

    all_kurtosis  = []
    all_delta_ppl = []
    plotted_models = set()
    plotted_quants = set()

    for qname, qstyle in quant_styles.items():
        for model, combos in summaries.items():
            kd = kv_dists.get(model)
            if kd is None:
                continue

            kurtosis   = kd["key_kurtosis_mean"]
            ctx_target = max(c["ctx"] for c in combos)
            row = next(
                (c for c in combos
                 if c["kv_quant"] == qname and c["ctx"] == ctx_target
                 and not c.get("asymmetric", False)),
                None
            )
            if row is None or row.get("ppl_delta") is None:
                continue

            delta_ppl      = abs(row["ppl_delta"])
            delta_ppl_plot = max(delta_ppl, 1e-4)

            color  = MODEL_COLORS[model]
            marker = MODEL_MARKERS[model]

            if qstyle["filled"]:
                ec, fc = "white", color
                lw = 0.8
            else:
                ec, fc = color, "none"
                lw = 1.5

            label = model if model not in plotted_models else None
            ax.scatter(kurtosis, delta_ppl_plot,
                       color=fc, marker=marker, s=qstyle["size"],
                       edgecolors=ec, linewidth=lw,
                       zorder=5, label=label)

            # Label all points; shift left for high-kurtosis models to stay in frame
            ha = "right" if kurtosis > 18 else "left"
            x_off = -8 if kurtosis > 18 else 8
            y_off = 6 if qstyle["filled"] else -12  # INT2 above, INT4 below

            # Per-point manual overrides to avoid overlap
            _key = (model, qname)
            _overrides = {
                ("Mistral-7B",  "int2-hqq"): dict(x_off=8,   y_off=-14, ha="left"),
                ("Qwen3-8B",    "int4-hqq"): dict(x_off=-8,  y_off=10,  ha="right"),
            }
            if _key in _overrides:
                ov = _overrides[_key]
                x_off, y_off, ha = ov["x_off"], ov["y_off"], ov["ha"]

            ax.annotate(model + qstyle["label_suffix"],
                        (kurtosis, delta_ppl_plot),
                        textcoords="offset points",
                        xytext=(x_off, y_off),
                        ha=ha, va="center",
                        fontsize=8, color=color, alpha=0.9)

            all_kurtosis.append(kurtosis)
            all_delta_ppl.append(np.log10(delta_ppl_plot))
            plotted_models.add(model)
            plotted_quants.add(qname)

    # ── Spearman correlation ───────────────────────────────────────────────────
    if len(all_kurtosis) >= 4:
        rho, pval = spearmanr(all_kurtosis, all_delta_ppl)
        pstr = f"p = {pval:.3f}" if pval >= 0.001 else "p < 0.001"
        ax.text(0.97, 0.05,
                f"Spearman ρ = {rho:.2f}\n{pstr}",
                transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    # ── Legend: quant-style indicator ─────────────────────────────────────────
    from matplotlib.lines import Line2D
    legend_quant = [
        Line2D([0], [0], marker="o", color="gray", markerfacecolor="gray",
               markersize=8, linewidth=0, label="INT2 (gefüllt)"),
        Line2D([0], [0], marker="o", color="gray", markerfacecolor="none",
               markersize=8, linewidth=0, markeredgewidth=1.5, label="INT4 (offen)"),
    ]
    # Model color/shape legend + quant style
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + legend_quant,
              labels=labels + ["INT2 (gefüllt)", "INT4 (offen)"],
              loc="upper left", framealpha=0.85, fontsize=9,
              ncol=1)

    ax.set_yscale("log")
    ax.set_xlabel("Key-Kurtosis (mean, alle Layers)", fontsize=11)
    ax.set_ylabel("|Δ-PPL| (log scale)", fontsize=11)
    ax.set_title("KV-Kurtosis als Prädiktor für Quantisierungs-Degradation\n"
                 "(höchste verfügbare Kontextlänge je Modell, WikiText-2)",
                 fontsize=12)

    # Risk-zone labels on right margin
    ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 1 else 25,
            0.003,  "verlustfrei", color="#2ca02c", fontsize=8, alpha=0.8,
            ha="right", va="center")
    ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 1 else 25,
            0.1,   "akzeptabel",  color="#ff7f0e", fontsize=8, alpha=0.8,
            ha="right", va="center")
    ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 1 else 25,
            100,   "kritisch",    color="#d62728", fontsize=8, alpha=0.8,
            ha="right", va="center")

    fig.tight_layout()
    save_fig(fig, "kurtosis_vs_ppl_delta.pdf", 1)
    plt.close(fig)


# ── Plot 2: VRAM vs Context Length ────────────────────────────────────────────

def plot_vram_vs_context(summaries):
    quants_show  = ["fp16", "int8-hqq", "int4-hqq", "int2-hqq"]
    quant_labels = {"fp16": "FP16", "int8-hqq": "INT8",
                    "int4-hqq": "INT4", "int2-hqq": "INT2"}
    quant_colors = {"fp16": "#4878CF", "int8-hqq": "#6ACC65",
                    "int4-hqq": "#D65F5F", "int2-hqq": "#B47CC7"}

    models = sorted(summaries.keys())
    # Build tick labels: "Model Name\n(ctx Xk)"
    def ctx_label(model):
        ctx = max(c["ctx"] for c in summaries[model])
        return f"{ctx // 1024}k" if ctx % 1024 == 0 else str(ctx)
    model_tick_labels = [f"{m}\n(ctx {ctx_label(m)})" for m in models]

    x      = np.arange(len(models))
    n_q    = len(quants_show)
    width  = 0.18
    offsets = np.linspace(-(n_q - 1) / 2, (n_q - 1) / 2, n_q) * width

    fig, (ax_abs, ax_rel) = plt.subplots(2, 1, figsize=(10, 8),
                                          gridspec_kw={"hspace": 0.45})

    # ── Top: absolute VRAM (GB) ───────────────────────────────────────────────
    for qi, q in enumerate(quants_show):
        vals = []
        for model in models:
            combos = summaries[model]
            ctx_target = max(c["ctx"] for c in combos)
            row = next((c for c in combos
                        if c["kv_quant"] == q and c["ctx"] == ctx_target), None)
            vals.append(row["vram_peak_mb"] / 1024 if row else np.nan)

        bars = ax_abs.bar(x + offsets[qi], vals, width,
                          color=quant_colors[q], label=quant_labels[q],
                          edgecolor="white", linewidth=0.5)
        # Value labels on bars
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax_abs.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.2,
                            f"{v:.1f}", ha="center", va="bottom",
                            fontsize=7, color="#333333")

    ax_abs.axhline(32.0, color="red", linestyle=":", linewidth=1.2,
                   label="32 GB limit", zorder=3)
    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels(model_tick_labels, rotation=15, ha="right", fontsize=9)
    ax_abs.set_ylabel("VRAM Peak (GB)")
    ax_abs.set_title("Absoluter VRAM-Verbrauch je Modell und Quantisierungsstufe",
                     fontsize=11)
    ax_abs.legend(fontsize=9, ncol=5, loc="upper left",
                   handlelength=1.2, handletextpad=0.4, columnspacing=0.8)
    ax_abs.set_ylim(0, 42)

    # ── Bottom: relative savings vs FP16 (%) ─────────────────────────────────
    for qi, q in enumerate(quants_show):
        if q == "fp16":
            continue
        savings = []
        for model in models:
            combos = summaries[model]
            ctx_target = max(c["ctx"] for c in combos)
            fp16_row = next((c for c in combos
                             if c["kv_quant"] == "fp16" and c["ctx"] == ctx_target), None)
            q_row    = next((c for c in combos
                             if c["kv_quant"] == q and c["ctx"] == ctx_target), None)
            if fp16_row and q_row:
                savings.append(
                    (1 - q_row["vram_peak_mb"] / fp16_row["vram_peak_mb"]) * 100
                )
            else:
                savings.append(np.nan)

        bars = ax_rel.bar(x + offsets[qi], savings, width,
                          color=quant_colors[q], label=quant_labels[q],
                          edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, savings):
            if not np.isnan(v):
                ax_rel.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.3,
                            f"{v:.1f}%", ha="center", va="bottom",
                            fontsize=7, color="#333333")

    ax_rel.set_xticks(x)
    ax_rel.set_xticklabels(model_tick_labels, rotation=15, ha="right", fontsize=9)
    ax_rel.set_ylabel("VRAM-Einsparung vs. FP16 (%)")
    ax_rel.set_title("Relative VRAM-Einsparung durch KV-Quantisierung", fontsize=11)
    ax_rel.legend(fontsize=9, ncol=3, loc="upper left",
                   handlelength=1.2, handletextpad=0.4, columnspacing=0.8)
    ax_rel.set_ylim(0, 35)

    fig.suptitle("VRAM-Analyse: absolut und relativ nach Quantisierungsstufe",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "vram_vs_context.pdf", 2)
    plt.close(fig)


# ── Plot 3: Δ-PPL Heatmap ─────────────────────────────────────────────────────

def plot_delta_ppl_heatmap(summaries):
    quants = ["int8-hqq", "int4-hqq", "int2-hqq", "int2-hqq(kivi)"]
    models = sorted(summaries.keys())

    # Build matrix; cap at 100 for display
    matrix = np.full((len(models), len(quants)), np.nan)
    for i, model in enumerate(models):
        combos = summaries[model]
        # Use highest available context for this model
        ctx_target = max(c["ctx"] for c in combos)
        for j, q in enumerate(quants):
            row = next(
                (c for c in combos
                 if c["kv_quant"] == q and c["ctx"] == ctx_target),
                None
            )
            if row and row.get("ppl_delta") is not None:
                matrix[i, j] = min(abs(row["ppl_delta"]), 100.0)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Use log-safe normalization
    from matplotlib.colors import LogNorm
    vmin = 1e-4
    vmax = 100.0
    # Fill NaN with 0 for display
    display = np.where(np.isnan(matrix), np.nan, np.clip(matrix, vmin, vmax))

    im = ax.imshow(display, aspect="auto",
                   norm=LogNorm(vmin=vmin, vmax=vmax),
                   cmap="RdYlGn_r")

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(quants)):
            val = matrix[i, j]
            if np.isnan(val):
                txt = "n/a"
                color = "gray"
            elif val < 0.01:
                txt = f"{val:.4f}"
                color = "black"
            elif val < 1.0:
                txt = f"{val:.3f}"
                color = "black"
            elif val < 10.0:
                txt = f"{val:.2f}"
                color = "white"
            else:
                txt = f"{val:.1f}"
                color = "white"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(range(len(quants)))
    quant_display = {"int8-hqq": "INT8", "int4-hqq": "INT4",
                     "int2-hqq": "INT2", "int2-hqq(kivi)": "INT2-KIVI"}
    ax.set_xticklabels([quant_display[q] for q in quants])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_title("|Δ-PPL| Heatmap (höchste Kontextlänge je Modell, WikiText-2)\n"
                 "Grün = verlustfrei, Rot = starke Degradation (log scale)",
                 fontsize=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("|Δ-PPL|", fontsize=10)

    fig.tight_layout()
    save_fig(fig, "delta_ppl_heatmap.pdf", 3)
    plt.close(fig)


# ── Plot 4: Needle-in-a-Haystack ──────────────────────────────────────────────

def plot_needle_comparison(summaries):
    # Load needle data directly from individual JSONs
    needle_data = {}  # model -> quant -> success_rate

    for f in glob.glob(str(LONG_CTX / "*.json")):
        if "summary" in f:
            continue
        d = json.load(open(f))
        model  = MODEL_LABELS.get(d["model"], d["model"])
        kv_cfg = d.get("kv_quant", {})
        bench  = d.get("benchmarks", {})
        needle = bench.get("needle_in_haystack", {})
        if not needle:
            continue

        if kv_cfg.get("enabled"):
            nbits   = kv_cfg.get("nbits", "?")
            backend = kv_cfg.get("backend", "")
            asym    = kv_cfg.get("axis_key", 1) == 0
            if asym:
                qname = f"int{nbits}-{backend}-kivi"
            else:
                qname = f"int{nbits}-{backend}"
        else:
            qname = "fp16"

        rate = needle.get("success_rate", 0.0)
        if model not in needle_data:
            needle_data[model] = {}
        needle_data[model][qname] = rate

    if not needle_data:
        print("  [4] No needle data found — skipping")
        return

    models = sorted(needle_data.keys())
    quants = [q for q in QUANT_ORDER if any(q in needle_data[m] for m in models)]

    x      = np.arange(len(models))
    n_q    = len(quants)
    width  = 0.7 / n_q

    fig, ax = plt.subplots(figsize=(13, 5))

    for j, q in enumerate(quants):
        vals = [needle_data[m].get(q, float("nan")) * 100 for m in models]
        offset = (j - n_q / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width,
                      color=QUANT_COLORS[q],
                      label=QUANT_LABELS[q],
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if not math.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1,
                        f"{v:.0f}%",
                        ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Needle Success Rate (%)")
    ax.set_title("Needle-in-a-Haystack: Retrieval-Erfolgsrate nach Modell & Quantisierung\n"
                 "(Alle Kontextlängen zusammengefasst, RULER-Noise Haystack)",
                 fontsize=12)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    fig.tight_layout()
    save_fig(fig, "needle_comparison.pdf", 4)
    plt.close(fig)


# ── Plot 5: Per-Layer Kurtosis Profile ────────────────────────────────────────

def plot_layer_kurtosis(kv_dists_raw):
    """Faceted heatmap: one subplot per model, own x-axis, shared log color scale."""
    from matplotlib.colors import LogNorm
    from matplotlib.cm import ScalarMappable

    models_sorted = sorted(
        kv_dists_raw.keys(),
        key=lambda m: kv_dists_raw[m]["summary"]["key_kurtosis_mean"]
    )
    n_models = len(models_sorted)

    # Shared colour scale across all models
    all_vals = [l["key"]["kurtosis"]
                for m in models_sorted
                for l in kv_dists_raw[m]["layers"]]
    norm = LogNorm(vmin=max(0.05, min(all_vals)), vmax=max(all_vals))

    fig, axes = plt.subplots(
        n_models, 1,
        figsize=(11, 1.35 * n_models + 1.2),
        gridspec_kw={"hspace": 0.18},
    )
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models_sorted):
        layers    = kv_dists_raw[model]["layers"]
        n         = len(layers)
        ys        = np.array([max(l["key"]["kurtosis"], 0.05) for l in layers])
        heat_row  = ys.reshape(1, -1)

        ax.imshow(
            heat_row,
            aspect="auto",
            norm=norm,
            cmap="YlOrRd",
            interpolation="bilinear",
            extent=[-0.5, n - 0.5, -0.5, 0.5],
        )

        # Peak marker
        peak_idx = int(np.argmax(ys))
        ax.plot(peak_idx, 0, marker="v", color="white",
                markersize=7, zorder=6,
                markeredgecolor="#555", markeredgewidth=0.7)
        ax.annotate(f"L{peak_idx}", xy=(peak_idx, 0),
                    xytext=(0, -12), textcoords="offset points",
                    ha="center", va="top", fontsize=7.5, color="#333")

        # Right-margin stats
        mean_k = kv_dists_raw[model]["summary"]["key_kurtosis_mean"]
        max_k  = kv_dists_raw[model]["summary"]["key_kurtosis_max"]
        ax.text(1.01, 0.5, f"ø {mean_k:.1f}  max {max_k:.0f}",
                transform=ax.transAxes,
                va="center", ha="left", fontsize=8.5, color="#333333")

        # y-axis: model label
        ax.set_yticks([0])
        ax.set_yticklabels([f"{model}\n(n={n})"], fontsize=9)
        ax.tick_params(axis="y", length=0)

        # x-axis: show ticks every 4 layers; only label bottom subplot
        tick_step = 4
        tick_pos  = list(range(0, n, tick_step))
        ax.set_xticks(tick_pos)
        if ax is axes[-1]:
            ax.set_xticklabels([str(t) for t in tick_pos], fontsize=8)
            ax.set_xlabel("Layer-Index (absolut)", fontsize=10)
        else:
            ax.set_xticklabels([])

        ax.set_xlim(-0.5, n - 0.5)

    # Shared colorbar on right side
    sm = ScalarMappable(cmap="YlOrRd", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.85, pad=0.12, aspect=30)
    cbar.set_label("Key-Kurtosis (log)", fontsize=9)

    fig.suptitle("KV-Cache Key-Kurtosis: Per-Layer-Profil aller Modelle\n"
                 "▼ = Peak-Layer (Lx = absoluter Layer-Index)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    save_fig(fig, "layer_kurtosis_profile.pdf", 5)
    plt.close(fig)


def plot_kurtosis_violin(kv_dists_raw):
    """Two-panel violin: low-kurtosis group (left) vs. high-kurtosis group (right)."""

    models_sorted = sorted(
        kv_dists_raw.keys(),
        key=lambda m: kv_dists_raw[m]["summary"]["key_kurtosis_mean"]
    )

    # Split by mean kurtosis threshold — Gemma/Mistral/Yi vs. Qwen2/Qwen3
    SPLIT_THRESHOLD = 15
    group_low  = [m for m in models_sorted
                  if kv_dists_raw[m]["summary"]["key_kurtosis_mean"] < SPLIT_THRESHOLD]
    group_high = [m for m in models_sorted
                  if kv_dists_raw[m]["summary"]["key_kurtosis_mean"] >= SPLIT_THRESHOLD]

    def _panel(ax, group, y_max, title_suffix):
        data = [[l["key"]["kurtosis"] for l in kv_dists_raw[m]["layers"]]
                for m in group]

        parts = ax.violinplot(data, positions=range(len(group)),
                              showmedians=True, showextrema=True, widths=0.6)
        for pc, m in zip(parts["bodies"], group):
            pc.set_facecolor(MODEL_COLORS[m])
            pc.set_alpha(0.65)
        for key in ("cmedians", "cmins", "cmaxes", "cbars"):
            parts[key].set_color("gray")
            parts[key].set_linewidth(0.9)

        rng = np.random.default_rng(42)
        for i, (m, vals) in enumerate(zip(group, data)):
            jitter = rng.uniform(-0.09, 0.09, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       color=MODEL_COLORS[m], s=22, alpha=0.55,
                       edgecolors="none", zorder=4)
            med = float(np.median(vals))
            mx  = max(vals)
            ax.text(i + 0.34, med, f"ø {med:.1f}", va="center",
                    fontsize=8.5, color=MODEL_COLORS[m])
            ax.text(i + 0.34, min(mx, y_max * 0.97),
                    f"max {mx:.0f}", va="top" if mx > y_max * 0.9 else "bottom",
                    fontsize=8.5, color=MODEL_COLORS[m])

        ax.set_ylim(-1, y_max)
        ax.set_xticks(range(len(group)))
        ax.set_xticklabels(group, rotation=20, ha="right", fontsize=10)
        ax.set_ylabel("Key-Kurtosis", fontsize=11)
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
        # Subtitle with y-range
        ax.set_title(f"{title_suffix}\n(y: 0–{y_max})", fontsize=10)

    # Compute y_max per group with a small margin
    y_max_low  = 20
    y_max_high = int(max(
        max(l["key"]["kurtosis"] for l in kv_dists_raw[m]["layers"])
        for m in group_high
    ) * 1.08)

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(9, 6),
        gridspec_kw={"width_ratios": [len(group_low), len(group_high)],
                     "wspace": 0.35}
    )

    _panel(ax_l, group_low,  y_max_low,  "Niedrige Kurtosis")
    _panel(ax_r, group_high, y_max_high, "Hohe Kurtosis (Ausreißer)")

    # Dividing visual separator
    fig.add_artist(plt.Line2D(
        [ax_l.get_position().x1 + 0.01,
         ax_l.get_position().x1 + 0.01],
        [0.1, 0.9],
        transform=fig.transFigure,
        color="lightgray", linewidth=1.2, linestyle="--"
    ))

    fig.suptitle("KV-Cache Key-Kurtosis: Verteilung je Layer (Modellvergleich)\n"
                 "Linkes Panel: homogene Modelle  |  Rechtes Panel: Heavy-Tail-Modelle",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    save_fig(fig, "kurtosis_violin.pdf", 6)
    plt.close(fig)


def plot_kv_key_distributions(kv_dists_raw):
    """Abbildung 7 — Key-Aktivierungsverteilungen: Linear (links) + Log-Skala (rechts).

    Zwei Panels zeigen die gemessenen Key-Histogramme vs. N(0,1)-Referenz.
    Log-Skala macht Schwanz-Unterschiede sichtbar: Gausssche Parabel vs. Heavy Tails.
    """
    from scipy.stats import norm as scipy_norm

    # Colors chosen for maximum contrast (overrides global MODEL_COLORS here)
    DIST_COLORS = {
        "Gemma-4-E4B": "#E63946",   # kräftiges Rot
        "Mistral-7B":  "#1D6FA4",   # Blau
        "Yi-1.5-9B":   "#2A9D8F",   # Teal
        "Qwen2-7B":    "#7B2D8B",   # Violett
        "Qwen3-8B":    "#F4830A",   # Orange
    }

    # Collect models that have histogram data
    models_with_hist = []
    for label, d in kv_dists_raw.items():
        layers = d.get("layers", [])
        if layers and "histogram" in layers[0].get("key", {}):
            models_with_hist.append(label)

    if not models_with_hist:
        print("  [plot_kv_key_distributions] No histogram data found — skipping. "
              "Re-run analyze_kv_distributions.py with --histogram-bins 200.")
        return

    # Sort by mean kurtosis (low → high) so Gemma appears first / lowest
    models_with_hist.sort(
        key=lambda m: kv_dists_raw[m]["summary"]["key_kurtosis_mean"]
    )

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(12, 4.5),
                                          gridspec_kw={"wspace": 0.28})

    x_ref = np.linspace(-6, 6, 600)
    gauss_pdf = scipy_norm.pdf(x_ref)

    for ax, yscale in [(ax_lin, "linear"), (ax_log, "log")]:
        # Reference Gaussian
        ax.plot(x_ref, gauss_pdf, color="black", lw=1.6,
                linestyle="--", label="N(0,1) Referenz", zorder=5)

        for label in models_with_hist:
            d = kv_dists_raw[label]
            layers = d["layers"]
            kurtosis_mean = d["summary"]["key_kurtosis_mean"]
            color = DIST_COLORS.get(label, "#888888")

            all_densities = np.array(
                [layer["key"]["histogram"]["density"] for layer in layers]
            )
            mean_density = all_densities.mean(axis=0)
            bin_centers = np.array(layers[0]["key"]["histogram"]["bin_centers"])

            ax.plot(
                bin_centers, mean_density,
                color=color, lw=1.9, alpha=0.82,
                label=f"{label}  (κ̄={kurtosis_mean:.1f})",
            )

        ax.set_xlim(-5.2, 5.2)
        ax.set_xlabel("z-normierter Key-Wert  (x − μ) / σ", fontsize=9.5)
        ax.tick_params(labelsize=8.5)

        if yscale == "log":
            ax.set_yscale("log")
            ax.set_ylim(5e-4, 2.0)
            ax.set_ylabel("Dichte (log)", fontsize=9.5)
            ax.set_title("Log-Skala — Schwanz-Verhalten", fontsize=10)
            # Annotate where heavy tails diverge from Gaussian
            ax.annotate(
                "Heavy Tails:\nModelle > N(0,1)",
                xy=(2.8, 8e-3), xytext=(3.5, 0.06),
                fontsize=7.5, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#888", lw=0.8),
                ha="center",
            )
        else:
            ax.set_ylim(bottom=0)
            ax.set_ylabel("Wahrscheinlichkeitsdichte", fontsize=9.5)
            ax.set_title("Lineare Skala — Peakform", fontsize=10)

    # Single shared legend on the right panel
    ax_log.legend(fontsize=8, framealpha=0.92, loc="lower center",
                  bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        "Key-Aktivierungsverteilungen im KV-Cache (alle Layer gemittelt, z-normiert)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    save_fig(fig, "kv_key_distributions.pdf", 7)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    summaries = load_summaries()
    kv_dists  = load_kv_dists()

    # Also load full kv_dist JSONs for layer-level plot
    kv_dists_raw = {}
    for f in glob.glob(str(KV_DIST / "*.json")):
        d = json.load(open(f))
        label = MODEL_LABELS.get(d["model"], d["model"])
        kv_dists_raw[label] = d

    print(f"  Models in summaries: {sorted(summaries.keys())}")
    print(f"  Models in kv_dists:  {sorted(kv_dists.keys())}")
    print()

    print("Generating plots...")
    plot_kurtosis_vs_ppl(summaries, kv_dists)
    plot_vram_vs_context(summaries)
    plot_delta_ppl_heatmap(summaries)
    plot_needle_comparison(summaries)
    plot_layer_kurtosis(kv_dists_raw)
    plot_kurtosis_violin(kv_dists_raw)
    plot_kv_key_distributions(kv_dists_raw)

    print()
    print(f"All plots saved to: {OUT_DIR}")
