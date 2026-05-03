#!/usr/bin/env python3
"""generate_thesis_plots.py

Generiert die 3 Thesis-PDFs, die in der Arbeit referenziert werden:

1) memory_vs_context_mistral.pdf
    KV-Cache Größe vs. Kontextlänge (Mistral-7B).

2) ppl_7b_comparison.pdf
    Absolute Perplexity bei 4096 Tokens (7B/8B Vergleich).
    Outlier werden geklippt und als Text annotiert.

3) quantizability_comparison.pdf
    ΔPPL (%) bei 4096 Tokens relativ zu FP16, als Funktion der GQA-Ratio.
    Outlier werden geklippt und als Text annotiert.

Unterstützte JSON-Formate:
- Schema v2.0 (quantize_kvcache_hf.py): config + measurements[].kv_cache.total_gb
- Profiling-Format (profile_*.json): model + measurements[].kv_cache_mb

Verwendung:
    python scripts/generate_thesis_plots.py --results-dir results/raw --output results/figures --max-context 4096
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# Paper-style formatting
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

PRECISION_COLORS = {
    'fp16': '#1f77b4',  # Blau
    'int8': '#2ca02c',  # Grün
    'int4': '#ff7f0e',  # Orange
    'int2': '#d62728',  # Rot
}

PRECISION_MARKERS = {
    'fp16': 'o',
    'int8': 's',
    'int4': '^',
    'int2': 'D',
}


MODEL_DISPLAY = {
    'mistralai/Mistral-7B-v0.1': 'Mistral-7B',
    'Qwen/Qwen3-8B': 'Qwen3-8B',
    'Qwen/Qwen2-7B': 'Qwen2-7B',
    '01-ai/Yi-1.5-9B': 'Yi-1.5-9B',
}

GQA_RATIOS = {
    'Mistral-7B': 4,
    'Qwen3-8B': 4,
    'Qwen2-7B': 7,
    'Yi-1.5-9B': 8,
}


def _delta_pct(ppl: float, baseline: float) -> float:
    return ((ppl - baseline) / baseline) * 100.0


def _clip(value: float, clip_max: float) -> tuple[float, bool]:
    if value is None or np.isnan(value):
        return 0.0, True
    if value > clip_max:
        return clip_max, True
    if value < -clip_max:
        return -clip_max, True
    return value, False


def _format_pct(value: float) -> str:
    if value is None:
        return 'n/a'
    abs_v = abs(value)
    if abs_v >= 1_000_000:
        return f"{value/1_000_000:.1f}M%"
    if abs_v >= 1_000:
        return f"{value/1_000:.1f}k%"
    if abs_v >= 100:
        return f"{value:.0f}%"
    if abs_v >= 10:
        return f"{value:.1f}%"
    return f"{value:.2f}%"


def _normalize_precision_backend_from_profile(config_name: str) -> tuple[str, str]:
    name = config_name.strip().lower()
    if name == 'fp16':
        return 'fp16', 'none'
    if 'int8' in name:
        return 'int8', 'hqq'
    if 'int4' in name:
        return 'int4', 'hqq'
    if 'int2' in name:
        return 'int2', 'hqq'
    return 'unknown', 'unknown'


def iter_records(results_dir: Path) -> list[dict]:
    """Load all supported JSONs into flat measurement records."""
    records: list[dict] = []

    for json_file in results_dir.glob('*.json'):
        try:
            data = json.loads(json_file.read_text())
        except Exception:
            continue

        # Schema v2.0 (quantize_kvcache_hf.py)
        if isinstance(data, dict) and 'config' in data and 'measurements' in data:
            cfg = data.get('config', {})
            model_id = cfg.get('model')
            if model_id not in MODEL_DISPLAY:
                continue
            model = MODEL_DISPLAY[model_id]
            precision = cfg.get('kv_precision', 'fp16')
            backend = cfg.get('backend', 'none')
            for m in data.get('measurements', []):
                try:
                    records.append({
                        'model': model,
                        'precision': precision,
                        'backend': backend,
                        'context': int(m['context_length']),
                        'ppl': float(m['perplexity']),
                        'kv_cache_gb': float(m['kv_cache']['total_gb']),
                        'kv_cache_mb': float(m['kv_cache']['total_gb']) * 1024,
                        'source': json_file.name,
                    })
                except Exception:
                    continue
            continue

        # Profiling format (profile_*.json)
        if isinstance(data, dict) and 'model' in data and 'measurements' in data and 'target_contexts' in data:
            model_id = data.get('model')
            if model_id not in MODEL_DISPLAY:
                continue
            model = MODEL_DISPLAY[model_id]
            for m in data.get('measurements', []):
                try:
                    precision, backend = _normalize_precision_backend_from_profile(m['config'])
                    records.append({
                        'model': model,
                        'precision': precision,
                        'backend': backend,
                        'context': int(m['context_len']),
                        'ppl': float(m['perplexity']),
                        'kv_cache_mb': float(m['kv_cache_mb']),
                        'kv_cache_gb': float(m['kv_cache_mb']) / 1024,
                        'source': json_file.name,
                    })
                except Exception:
                    continue
            continue

    return records


def _pick_value_at_context(records: list[dict], context: int) -> dict | None:
    """Pick the record at exact context if present, else nearest lower, else nearest."""
    if not records:
        return None
    exact = [r for r in records if r['context'] == context]
    if exact:
        return exact[0]
    lower = [r for r in records if r['context'] < context]
    if lower:
        return max(lower, key=lambda r: r['context'])
    return min(records, key=lambda r: abs(r['context'] - context))


def plot_memory_vs_context_mistral(results_dir: Path, output_path: Path, max_context: int):
    """Plot 1: KV-Cache Speicher vs. Kontextlänge für Mistral-7B."""
    
    print("\n1️⃣  Generiere memory_vs_context_mistral.pdf...")
    
    records = iter_records(results_dir)
    mistral = [r for r in records if r['model'] == 'Mistral-7B']
    if not mistral:
        print("  ⚠️  Keine Mistral-Daten gefunden!")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Pro Precision eine Linie plotten (bevorzuge HQQ wenn mehrere Backends existieren)
    for precision in ['fp16', 'int8', 'int4', 'int2']:
        subset = [r for r in mistral if r['precision'] == precision]
        if not subset:
            continue
        # Prefer hqq over others
        hqq = [r for r in subset if r['backend'] == 'hqq']
        use = hqq if hqq else subset
        use = [r for r in use if r['context'] <= max_context]
        use = sorted(use, key=lambda r: r['context'])
        if not use:
            continue
        contexts = [r['context'] for r in use]
        kv_gb = [r['kv_cache_gb'] for r in use]
        label = f"{precision.upper()}"
        ax.plot(
            contexts,
            kv_gb,
            marker=PRECISION_MARKERS[precision],
            color=PRECISION_COLORS[precision],
            linewidth=2,
            markersize=7,
            label=label,
            alpha=0.9,
        )
    
    ax.set_xlabel('Context Length (tokens)')
    ax.set_ylabel('KV-Cache Size (GB)')
    ax.set_title(f'Mistral-7B: KV-Cache Memory vs. Context Length (≤ {max_context})')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # X-Achse: Zweierpotenzen (tatsächliche Messpunkte)
    context_ticks = [512, 1024, 2048, 4096]
    if max_context >= 8192:
        context_ticks.append(8192)
    if max_context >= 16384:
        context_ticks.append(16384)
    ax.set_xticks(context_ticks)
    ax.set_xticklabels([f'{c//1024}k' if c >= 1024 else str(c) for c in context_ticks])
    ax.set_xlim(left=0, right=max_context + 500)
    
    # Y-Achse: schönere Ticks für GB-Werte
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6, integer=False))
    
    plt.tight_layout()
    plt.savefig(output_path / 'memory_vs_context_mistral.pdf')
    plt.close()
    
    print(f"  ✅ Gespeichert: {output_path / 'memory_vs_context_mistral.pdf'}")


def plot_ppl_7b_comparison(results_dir: Path, output_path: Path, max_context: int):
    """Plot 2: Stability Heatmap — kompakte Darstellung der Quantisierungsrobustheit.
    
    Zeigt sofort: INT8 immer ok, INT4 abhängig von GQA, INT2 kritisch.
    """
    
    print("\n2️⃣  Generiere stability_heatmap.pdf...")
    
    records = iter_records(results_dir)
    model_order = ['Mistral-7B', 'Qwen3-8B', 'Qwen2-7B', 'Yi-1.5-9B']
    available = [m for m in model_order if any(r['model'] == m for r in records)]
    
    if not available:
        print("  ⚠️  Keine Modell-Daten gefunden!")
        return

    # Compute ΔPPL matrix
    precisions = ['int8', 'int4', 'int2']
    matrix = np.zeros((len(available), len(precisions)))
    matrix[:] = np.nan
    
    for i, model in enumerate(available):
        fp16 = [r for r in records if r['model'] == model and r['precision'] == 'fp16']
        fp16_pick = _pick_value_at_context(fp16, max_context)
        if fp16_pick is None:
            continue
        baseline = float(fp16_pick['ppl'])
        
        for j, precision in enumerate(precisions):
            subset = [r for r in records if r['model'] == model and r['precision'] == precision]
            if not subset:
                continue
            hqq = [r for r in subset if r['backend'] == 'hqq']
            use = hqq if hqq else subset
            pick = _pick_value_at_context(use, max_context)
            if pick is None:
                continue
            delta = _delta_pct(float(pick['ppl']), baseline)
            matrix[i, j] = delta
    
    # Plot heatmap mit Farbcodierung
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    # Custom colormap: grün <1%, gelb <10%, orange <100%, rot ≥100%
    from matplotlib.colors import ListedColormap, BoundaryNorm
    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']  # grün, gelb, orange, rot
    bounds = [0, 1, 10, 100, 1e6]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Clip extreme values für bessere Darstellung
    matrix_plot = np.clip(matrix, 0, 1e6)
    
    im = ax.imshow(matrix_plot, cmap=cmap, norm=norm, aspect='auto')
    
    # Annotations mit Werten
    # Werte unter 1% werden auf "< 1%" vereinheitlicht (Messgenauigkeit rechtfertigt keine höhere Präzision)
    for i in range(len(available)):
        for j in range(len(precisions)):
            val = matrix[i, j]
            if np.isnan(val):
                text = 'N/A'
                color = 'gray'
            elif val >= 100:
                text = 'BROKEN'
                color = 'white'
            elif val >= 10:
                text = f'{val:.1f}%'
                color = 'black'
            elif val >= 1:
                text = f'{val:.1f}%'
                color = 'black'
            elif abs(val) < 1:
                # Werte zwischen -1% und +1% sind innerhalb der Messungenauigkeit
                text = '< 1%'
                color = 'black'
            else:
                text = f'{val:.1f}%'
                color = 'black'
            
            ax.text(j, i, text, ha='center', va='center', 
                   fontsize=9, weight='bold', color=color)
    
    # Achsenbeschriftung
    ax.set_xticks(np.arange(len(precisions)))
    ax.set_yticks(np.arange(len(available)))
    ax.set_xticklabels([p.upper() for p in precisions])
    ax.set_yticklabels([f"{m} (GQA {GQA_RATIOS.get(m, '?')}:1)" for m in available])
    
    ax.set_xlabel('KV-Cache Precision', fontsize=11)
    ax.set_ylabel('Model Architecture', fontsize=11)
    ax.set_title(f'Quantization Stability at {max_context} Tokens (ΔPPL vs FP16)', fontsize=12)
    
    # Colorbar mit Labels
    cbar = plt.colorbar(im, ax=ax, ticks=[0.5, 5, 50, 500])
    cbar.set_label('ΔPPL (%)', fontsize=10)
    cbar.ax.set_yticklabels(['<1%\nStable', '<10%\nAcceptable', '<100%\nDegraded', '≥100%\nBroken'])
    
    plt.tight_layout()
    plt.savefig(output_path / 'stability_heatmap.pdf')
    plt.close()
    
    print(f"  ✅ Gespeichert: {output_path / 'stability_heatmap.pdf'}")


def plot_quantizability_comparison(results_dir: Path, output_path: Path):
    """Plot 3: GQA-Ratio vs. INT4 Degradation — zeigt den Mechanismus klar.
    
    Fokus nur auf INT4 (weil INT2 zu chaotisch), 4 Punkte, klarer Trend.
    """
    
    print("\n3️⃣  Generiere gqa_mechanism.pdf...")
    
    records = iter_records(results_dir)
    model_order = ['Mistral-7B', 'Qwen3-8B', 'Qwen2-7B', 'Yi-1.5-9B']
    available = [m for m in model_order if any(r['model'] == m for r in records) and m in GQA_RATIOS]
    
    if not available:
        print("  ⚠️  Keine Modelle mit GQA-Ratios gefunden!")
        return
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Nur INT4 für cleanen Plot
    points = []
    for model in available:
        gqa = GQA_RATIOS.get(model)
        if gqa is None:
            continue
        
        fp16 = [r for r in records if r['model'] == model and r['precision'] == 'fp16']
        fp16_pick = _pick_value_at_context(fp16, 4096)
        if fp16_pick is None:
            continue
        baseline = float(fp16_pick['ppl'])
        
        # INT4 only
        subset = [r for r in records if r['model'] == model and r['precision'] == 'int4']
        if not subset:
            continue
        hqq = [r for r in subset if r['backend'] == 'hqq']
        use = hqq if hqq else subset
        pick = _pick_value_at_context(use, 4096)
        if pick is None:
            continue
        
        delta = _delta_pct(float(pick['ppl']), baseline)
        points.append((gqa, delta, model))
    
    if not points:
        print("  ⚠️  Keine INT4-Daten für GQA-Analyse!")
        return
    
    # Scatter plot
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    ax.scatter(xs, ys, s=150, marker='o', 
               color=PRECISION_COLORS['int4'], alpha=0.9,
               edgecolors='black', linewidths=1.5, zorder=3)
    
    # Labels nur für auffällige Punkte
    for x, y, model in points:
        if y > 10 or model == 'Mistral-7B':  # Nur Outlier + Referenz labeln
            ax.annotate(model, (x, y), textcoords='offset points',
                       xytext=(0, 10), ha='center', fontsize=9, weight='bold')
    
    # Schwellenlinien
    ax.axhline(y=1, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='1% threshold (stable)')
    ax.axhline(y=10, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='10% threshold (degraded)')
    ax.axhline(y=100, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='100% threshold (broken)')
    
    ax.set_xlabel('GQA Ratio (queries per KV-head)', fontsize=12)
    ax.set_ylabel('ΔPPL vs FP16 at 4096 tokens (%)', fontsize=12)
    ax.set_title('INT4 KV-Cache Quantization Sensitivity vs. GQA Architecture', fontsize=13)
    ax.set_xticks(sorted(set(GQA_RATIOS[m] for m in available)))
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_yscale('log')
    ax.set_ylim(0.1, max(ys) * 3)
    ax.set_xlim(3, 9)
    
    plt.tight_layout()
    plt.savefig(output_path / 'gqa_mechanism.pdf')
    plt.close()
    
    print(f"  ✅ Gespeichert: {output_path / 'gqa_mechanism.pdf'}")


def main():
    parser = argparse.ArgumentParser(description="Generiert Thesis-Plots")
    parser.add_argument("--results-dir", type=str, default="results/raw",
                       help="Verzeichnis mit JSON-Ergebnissen")
    parser.add_argument("--output", type=str, default="results/figures",
                       help="Output-Verzeichnis für Plots")
    parser.add_argument("--max-context", type=int, default=4096,
                       help="Maximale Kontextlänge für vergleichbare Plots (Default: 4096)")
    
    args = parser.parse_args()
    
    # Pfade
    if Path(args.results_dir).is_absolute():
        results_dir = Path(args.results_dir)
    else:
        results_dir = Path.cwd() / args.results_dir
    
    if Path(args.output).is_absolute():
        output_path = Path(args.output)
    else:
        output_path = Path.cwd() / args.output
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 Generiere Thesis-Plots aus: {results_dir}")
    print(f"📁 Output: {output_path}")
    print(f"🔧 max_context = {args.max_context}\n")
    
    # Plots generieren
    plot_memory_vs_context_mistral(results_dir, output_path, args.max_context)
    plot_ppl_7b_comparison(results_dir, output_path, args.max_context)
    plot_quantizability_comparison(results_dir, output_path)
    
    print("\n✅ Alle Plots generiert!\n")
    print("Hinweis: pareto_main.pdf wird separat mit plot_pareto.py erstellt:")
    print("  python scripts/plot_pareto.py --results-dir results/raw --output results/figures/pareto_main.pdf")


if __name__ == "__main__":
    main()
