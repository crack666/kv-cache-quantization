#!/usr/bin/env python3
"""
Pareto-Frontier Visualization für KV-Cache Quantisierung

Erstellt Plots im Stil von KIVI/ATOM Papers:
- X-Achse: Compression Ratio (1× bis 8×)
- Y-Achse: PPL Degradation (%)
- Marker: Modell × Backend

Verwendung:
    python plot_pareto.py --results-dir ../results/raw --output pareto.pdf
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Stil-Konfiguration (Paper-ready)
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Modell-spezifische Farben
MODEL_COLORS = {
    'gpt2': '#1f77b4',           # Blau
    'Qwen2-0.5B': '#ff7f0e',     # Orange
    'Qwen2-7B': '#d62728',       # Rot
    'Mistral-7B': '#2ca02c',     # Grün
    'Qwen3-8B': '#9467bd',       # Lila
    'Yi-1.5-9B': '#8c564b',      # Braun
}

# Backend-Marker
BACKEND_MARKERS = {
    'fp16': 'o',      # Kreis (Baseline)
    'hqq': 's',       # Quadrat
    'quanto': '^',    # Dreieck
}

# Precision-Linestyles
PRECISION_SIZES = {
    'fp16': 150,
    'int8': 120,
    'int4': 100,
    'int2': 80,
}


def normalize_model_name(name: str) -> str:
    """Normalisiert Modellnamen für konsistente Darstellung."""
    if 'gpt2' in name.lower():
        return 'gpt2'
    elif 'qwen2-0.5b' in name.lower() or 'Qwen_Qwen2-0.5B' in name:
        return 'Qwen2-0.5B'
    elif 'qwen2-7b' in name.lower() or 'Qwen_Qwen2-7B' in name:
        return 'Qwen2-7B'
    elif 'qwen3-8b' in name.lower() or 'Qwen_Qwen3-8B' in name or 'Qwen2.5-Coder-7B' in name:
        return 'Qwen3-8B'
    elif 'yi-1.5-9b' in name.lower() or '01-ai_Yi-1.5-9B' in name:
        return 'Yi-1.5-9B'
    elif 'mistral' in name.lower():
        return 'Mistral-7B'
    return name


def load_results(results_dir: Path) -> list[dict]:
    """Lädt alle JSON-Ergebnisse aus dem Verzeichnis."""
    results = []
    
    for json_file in results_dir.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            # Schema v2.0 check
            if 'config' not in data:
                continue
            
            config = data['config']
            summary = data.get('summary', {})
            
            # PPL extrahieren
            ppl = summary.get('avg_ppl') or summary.get('avg_perplexity')
            if ppl is None:
                continue
            
            result = {
                'file': json_file.name,
                'model': normalize_model_name(config.get('model', '')),
                'precision': config.get('kv_precision', 'fp16'),
                'nbits': config.get('nbits'),
                'backend': config.get('backend', 'none'),
                'ppl': ppl,
                'compression': 1.0,  # Wird später berechnet
            }
            
            results.append(result)
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warnung: {json_file.name} - {e}")
    
    return results


def calculate_compression_and_delta(results: list[dict]) -> list[dict]:
    """Berechnet Kompressionsrate und PPL-Delta pro Modell."""
    
    # FP16 Baselines pro Modell finden (nimm den niedrigsten PPL)
    baselines = {}
    for r in results:
        if r['precision'] == 'fp16':
            model = r['model']
            if model not in baselines or r['ppl'] < baselines[model]:
                baselines[model] = r['ppl']
    
    # Kompression und Delta berechnen
    for r in results:
        model = r['model']
        baseline_ppl = baselines.get(model, r['ppl'])
        
        # PPL Delta
        r['ppl_delta_pct'] = ((r['ppl'] - baseline_ppl) / baseline_ppl) * 100
        
        # Kompression basierend auf Precision und Backend
        precision = r['precision']
        backend = r['backend']
        
        if precision == 'fp16':
            r['compression'] = 1.0
        elif precision == 'int8':
            r['compression'] = 2.0
        elif precision == 'int4':
            # HQQ packt echte 4-bit, quanto hat overhead
            r['compression'] = 4.0 if backend == 'hqq' else 2.0
        elif precision == 'int2':
            r['compression'] = 8.0 if backend == 'hqq' else 2.0
    
    return results


def filter_best_per_config(results: list[dict]) -> list[dict]:
    """Wählt das beste Ergebnis pro Modell+Precision+Backend."""
    best = {}
    
    for r in results:
        key = (r['model'], r['precision'], r['backend'])
        if key not in best or r['ppl'] < best[key]['ppl']:
            best[key] = r
    
    return list(best.values())


def plot_pareto_all_models(results: list[dict], output_path: Path, log_scale: bool = False, no_show: bool = False):
    """Erstellt Pareto-Plot für alle Modelle."""
    
    fig, ax = plt.subplots()
    
    # Plot pro Modell
    for model, color in MODEL_COLORS.items():
        model_results = [r for r in results if r['model'] == model]
        
        for r in model_results:
            backend = r['backend'] if r['backend'] != 'none' else 'fp16'
            marker = BACKEND_MARKERS.get(backend, 'o')
            size = PRECISION_SIZES.get(r['precision'], 100)
            
            # Outlier-Check (Qwen explodiert)
            ppl_delta = r['ppl_delta_pct']
            if ppl_delta > 100:
                # Marker am oberen Rand mit Pfeil nach oben
                ax.scatter(r['compression'], 100, c=color, marker=marker, s=size,
                          edgecolors='black', linewidths=0.5, alpha=0.7)
                ax.annotate(f"{ppl_delta:.0f}%↑", (r['compression'], 100),
                           textcoords="offset points", xytext=(0, 5),
                           ha='center', fontsize=7, color=color)
            else:
                ax.scatter(r['compression'], ppl_delta, c=color, marker=marker, s=size,
                          edgecolors='black', linewidths=0.5, alpha=0.9,
                          label=f"{model} {r['precision']}")
    
    # Pareto-optimale Punkte hervorheben
    pareto_front = find_pareto_front(results)
    for r in pareto_front:
        if r['ppl_delta_pct'] <= 100:  # Nur sinnvolle
            ax.scatter(r['compression'], r['ppl_delta_pct'], 
                      facecolors='none', edgecolors='gold', 
                      linewidths=2, s=200, marker='o', zorder=5)
    
    # Achsen
    ax.set_xlabel('Compression Ratio (×)')
    ax.set_ylabel('PPL Degradation (%)')
    ax.set_xlim(0.5, 9)
    ax.set_ylim(-5, 105)  # 100% als praktisches Maximum
    
    # Referenzlinien
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(y=1, color='green', linestyle=':', linewidth=1, alpha=0.5, label='1% threshold')
    ax.axhline(y=10, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='10% threshold')
    
    # Legende
    legend_handles = [
        mpatches.Patch(color=color, label=model) 
        for model, color in MODEL_COLORS.items()
    ]
    legend_handles.extend([
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=8, label='FP16 (baseline)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                   markersize=8, label='HQQ backend'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
                   markersize=8, label='Quanto backend'),
        plt.Line2D([0], [0], marker='o', color='w', markeredgecolor='gold',
                   markersize=10, markeredgewidth=2, label='Pareto-optimal'),
    ])
    
    ax.legend(handles=legend_handles, loc='upper left', ncol=2)
    
    # Titel
    ax.set_title('KV-Cache Quantization: Compression vs. Quality Trade-off')
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Speichern
    plt.savefig(output_path, format=output_path.suffix[1:])
    print(f"Plot gespeichert: {output_path}")
    
    plt.close()
    
    return fig


def plot_pareto_per_model(results: list[dict], output_dir: Path):
    """Erstellt separate Pareto-Plots pro Modell."""
    
    for model in MODEL_COLORS.keys():
        model_results = [r for r in results if r['model'] == model]
        if not model_results:
            continue
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Group by backend
        for backend, marker in BACKEND_MARKERS.items():
            backend_results = [r for r in model_results if 
                             (r['backend'] == backend) or 
                             (backend == 'fp16' and r['precision'] == 'fp16')]
            
            if not backend_results:
                continue
            
            compressions = [r['compression'] for r in backend_results]
            deltas = [min(r['ppl_delta_pct'], 100) for r in backend_results]  # Clip at 100
            
            ax.scatter(compressions, deltas, marker=marker, s=100,
                      c=MODEL_COLORS[model], edgecolors='black', linewidths=0.5,
                      label=backend.upper(), alpha=0.8)
        
        ax.set_xlabel('Compression Ratio (×)')
        ax.set_ylabel('PPL Degradation (%)')
        ax.set_title(f'{model}: KV-Cache Quantization Pareto')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, 9)
        
        # Y-Limit anpassen
        max_delta = max(min(r['ppl_delta_pct'], 100) for r in model_results)
        ax.set_ylim(-2, max(max_delta * 1.2, 25))
        
        output_path = output_dir / f"pareto_{model.lower().replace('-', '_')}.pdf"
        plt.savefig(output_path)
        print(f"Plot gespeichert: {output_path}")
        plt.close()


def find_pareto_front(results: list[dict]) -> list[dict]:
    """Findet die Pareto-optimalen Punkte."""
    pareto = []
    
    # Nur sinnvolle Ergebnisse
    valid = [r for r in results if r['ppl_delta_pct'] < 50]
    
    for r in valid:
        dominated = False
        for other in valid:
            # other dominiert r wenn: mehr Kompression UND weniger Degradation
            if (other['compression'] >= r['compression'] and 
                other['ppl_delta_pct'] <= r['ppl_delta_pct'] and
                (other['compression'] > r['compression'] or 
                 other['ppl_delta_pct'] < r['ppl_delta_pct'])):
                dominated = True
                break
        
        if not dominated:
            pareto.append(r)
    
    return pareto


def print_summary_table(results: list[dict]):
    """Gibt eine Zusammenfassungstabelle aus."""
    
    print("\n" + "=" * 80)
    print("ERGEBNIS-ZUSAMMENFASSUNG")
    print("=" * 80)
    
    # Sortiert nach Modell, dann Kompression
    sorted_results = sorted(results, key=lambda x: (x['model'], x['compression']))
    
    print(f"\n{'Modell':<15} {'Precision':<8} {'Backend':<8} {'Kompr.':<8} {'PPL':<10} {'Δ PPL':<12}")
    print("-" * 70)
    
    for r in sorted_results:
        delta_str = f"+{r['ppl_delta_pct']:.1f}%" if r['ppl_delta_pct'] >= 0 else f"{r['ppl_delta_pct']:.1f}%"
        
        # Status-Icon
        if r['ppl_delta_pct'] > 100:
            icon = "❌"
        elif r['ppl_delta_pct'] > 10:
            icon = "⚠️"
        elif r['ppl_delta_pct'] > 1:
            icon = "🟡"
        else:
            icon = "✅"
        
        print(f"{r['model']:<15} {r['precision']:<8} {r['backend']:<8} {r['compression']:.1f}×     {r['ppl']:<10.3f} {delta_str:<10} {icon}")
    
    # Pareto-Front
    print("\n" + "-" * 70)
    print("PARETO-OPTIMALE KONFIGURATIONEN:")
    print("-" * 70)
    
    pareto = find_pareto_front(results)
    for r in sorted(pareto, key=lambda x: x['compression']):
        print(f"  🏆 {r['model']} {r['precision'].upper()}-{r['backend']}: "
              f"{r['compression']:.0f}× Kompression, +{r['ppl_delta_pct']:.1f}% PPL")


def main():
    parser = argparse.ArgumentParser(description="Pareto-Plot für KV-Cache Quantisierung")
    parser.add_argument("--results-dir", type=str, default="../results/raw",
                       help="Verzeichnis mit JSON-Ergebnissen")
    parser.add_argument("--output", type=str, default="../results/figures/pareto_all.pdf",
                       help="Output-Pfad für den Plot")
    parser.add_argument("--per-model", action="store_true",
                       help="Zusätzlich separate Plots pro Modell erstellen")
    parser.add_argument("--log-scale", action="store_true",
                       help="Logarithmische Y-Achse (für Outlier)")
    parser.add_argument("--no-show", action="store_true",
                       help="Kein interaktives Fenster öffnen")
    
    args = parser.parse_args()
    
    # Pfade
    script_dir = Path(__file__).parent
    # Wenn Pfad relativ ist, von CWD auflösen, nicht von script_dir
    if Path(args.results_dir).is_absolute():
        results_dir = Path(args.results_dir).resolve()
    else:
        results_dir = Path.cwd() / args.results_dir
    
    if Path(args.output).is_absolute():
        output_path = Path(args.output).resolve()
    else:
        output_path = Path.cwd() / args.output
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Lade Ergebnisse aus: {results_dir}")
    
    # Daten laden und verarbeiten
    results = load_results(results_dir)
    print(f"  {len(results)} Experimente geladen")
    
    results = calculate_compression_and_delta(results)
    results = filter_best_per_config(results)
    print(f"  {len(results)} beste Konfigurationen")
    
    # Zusammenfassung
    print_summary_table(results)
    
    # Plots erstellen
    plot_pareto_all_models(results, output_path, args.log_scale)
    
    if args.per_model:
        plot_pareto_per_model(results, output_path.parent)
    
    print("\n✅ Fertig!")


if __name__ == "__main__":
    main()
