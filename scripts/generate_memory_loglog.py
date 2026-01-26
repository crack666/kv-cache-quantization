#!/usr/bin/env python3
"""Test: Log-Log-Plot für Memory vs Context (Mistral-7B)."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

PRECISION_COLORS = {
    'fp16': '#1f77b4',
    'int8': '#2ca02c',
    'int4': '#ff7f0e',
    'int2': '#d62728',
}

PRECISION_MARKERS = {
    'fp16': 'o',
    'int8': 's',
    'int4': '^',
    'int2': 'D',
}

MODEL_DISPLAY = {
    'mistralai/Mistral-7B-v0.1': 'Mistral-7B',
}

def load_mistral_data(results_dir: Path):
    """Load all Mistral measurements from profile_*.json format."""
    records = []
    
    for json_file in results_dir.glob('profile_mistral*.json'):
        try:
            data = json.loads(json_file.read_text())
        except Exception:
            continue
        
        # New format: {'model': 'mistralai/Mistral-7B-v0.1', 'measurements': [...]}
        if isinstance(data, dict) and 'measurements' in data:
            model_id = data.get('model', '')
            if 'Mistral' not in model_id:
                continue
            
            for m in data.get('measurements', []):
                try:
                    config = m.get('config', 'FP16')
                    # Map config names to precision
                    if config == 'FP16':
                        precision = 'fp16'
                    elif 'INT8' in config:
                        precision = 'int8'
                    elif 'INT4' in config:
                        precision = 'int4'
                    elif 'INT2' in config:
                        precision = 'int2'
                    else:
                        continue
                    
                    records.append({
                        'precision': precision,
                        'backend': 'hqq' if 'HQQ' in config else 'none',
                        'context': int(m['context_len']),
                        'kv_cache_gb': float(m['kv_cache_mb']) / 1024,
                        'kv_cache_mb': float(m['kv_cache_mb']),
                    })
                except Exception:
                    continue
    
    return records

def main():
    results_dir = Path('results/raw')
    output_path = Path('results/figures')
    output_path.mkdir(parents=True, exist_ok=True)
    
    records = load_mistral_data(results_dir)
    if not records:
        print("⚠️  Keine Mistral-Daten gefunden!")
        return
    
    # Teste beide: GB und MB
    for unit, factor, ylabel in [('GB', 1, 'KV-Cache Size (GB)'), ('MB', 1024, 'KV-Cache Size (MB)')]:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for precision in ['fp16', 'int8', 'int4', 'int2']:
            subset = [r for r in records if r['precision'] == precision]
            if not subset:
                continue
            
            # Prefer HQQ
            hqq = [r for r in subset if r['backend'] == 'hqq']
            use = hqq if hqq else subset
            use = sorted(use, key=lambda r: r['context'])
            
            contexts = [r['context'] for r in use]
            if unit == 'GB':
                sizes = [r['kv_cache_gb'] for r in use]
            else:
                sizes = [r['kv_cache_mb'] for r in use]
            
            ax.plot(
                contexts,
                sizes,
                marker=PRECISION_MARKERS[precision],
                color=PRECISION_COLORS[precision],
                linewidth=2,
                markersize=8,
                label=precision.upper(),
                alpha=0.9,
            )
        
        ax.set_xlabel('Context Length (tokens)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title('Mistral-7B: KV-Cache Memory Scaling (Log-Log)', fontsize=13)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)
        
        # Log-Log Skala
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=2)
        
        # X-Achse: Zweierpotenzen (nur bis 4k, da wir nur bis 4096 messen)
        context_ticks = [128, 256, 512, 1024, 2048, 4096]
        ax.set_xticks(context_ticks)
        ax.set_xticklabels([f'{c//1024}k' if c >= 1024 else str(c) for c in context_ticks])
        
        # Y-Achse: angepasst an tatsächliche Werte
        if unit == 'GB':
            # Typische Werte: INT2 ~0.03-0.27 GB, FP16 ~0.06-2.15 GB
            y_ticks = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0]
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([f'{y:.3g}' for y in y_ticks])
        else:
            # MB: INT2 ~32-268 MB, FP16 ~64-2150 MB
            y_ticks = [32, 64, 128, 256, 512, 1024, 2048]
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([f'{y}' for y in y_ticks])
        
        plt.tight_layout()
        filename = f'memory_loglog_test_{unit.lower()}.pdf'
        plt.savefig(output_path / filename)
        plt.close()
        
        print(f"✅ Gespeichert: {output_path / filename}")

if __name__ == "__main__":
    main()
