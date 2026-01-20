#!/usr/bin/env python3
"""Analysiert die tatsächlichen ΔPPL-Werte bei 4096 aus allen Raw-JSONs."""

import json
from pathlib import Path
from collections import defaultdict

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

def norm_profile_cfg(name):
    name = name.strip().lower()
    if name == 'fp16':
        return 'fp16', 'none'
    if 'int8' in name:
        return 'int8', 'hqq'
    if 'int4' in name:
        return 'int4', 'hqq'
    if 'int2' in name:
        return 'int2', 'hqq'
    return None, None

def pick_at(recs, ctx):
    exact = [r for r in recs if r['context'] == ctx]
    if exact:
        return exact[0]
    lower = [r for r in recs if r['context'] < ctx]
    if lower:
        return max(lower, key=lambda r: r['context'])
    return None

def main():
    raw = Path('results/raw')
    rows = []
    
    # Load all JSONs
    for p in raw.glob('*.json'):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        
        # Schema v2.0
        if isinstance(data, dict) and 'config' in data and 'measurements' in data:
            cfg = data['config']
            m = MODEL_DISPLAY.get(cfg.get('model'))
            if not m:
                continue
            prec = cfg.get('kv_precision', 'fp16')
            backend = cfg.get('backend', 'none')
            for meas in data.get('measurements', []):
                rows.append({
                    'model': m,
                    'precision': prec,
                    'backend': backend,
                    'context': int(meas['context_length']),
                    'ppl': float(meas['perplexity'])
                })
        
        # Profiling format
        elif isinstance(data, dict) and 'model' in data and 'measurements' in data and 'target_contexts' in data:
            m = MODEL_DISPLAY.get(data.get('model'))
            if not m:
                continue
            for meas in data.get('measurements', []):
                prec, backend = norm_profile_cfg(meas['config'])
                if not prec:
                    continue
                rows.append({
                    'model': m,
                    'precision': prec,
                    'backend': backend,
                    'context': int(meas['context_len']),
                    'ppl': float(meas['perplexity'])
                })
    
    ctx = 4096
    models = sorted(set(r['model'] for r in rows))
    
    print(f"\n{'='*80}")
    print(f"ΔPPL-Analyse bei {ctx} Tokens (preferring HQQ backend)")
    print(f"{'='*80}\n")
    
    # Tabelle für die Thesis
    print(f"{'Model':<15} {'GQA':<4} {'FP16':<10} {'INT8':<15} {'INT4':<15} {'INT2':<15}")
    print("-" * 80)
    
    for model in models:
        gqa = GQA_RATIOS.get(model, '?')
        
        # FP16 baseline
        base_recs = [r for r in rows if r['model'] == model and r['precision'] == 'fp16']
        base = pick_at(base_recs, ctx)
        if not base:
            print(f"{model:<15} {gqa:<4} {'N/A':<10}")
            continue
        
        base_ppl = base['ppl']
        row = [f"{model:<15}", f"{gqa:<4}", f"{base_ppl:.4f}"]
        
        for prec in ['int8', 'int4', 'int2']:
            recs = [r for r in rows if r['model'] == model and r['precision'] == prec]
            if not recs:
                row.append(f"{'N/A':<15}")
                continue
            
            # Prefer HQQ
            hqq = [r for r in recs if r['backend'] == 'hqq']
            use = hqq if hqq else recs
            pick = pick_at(use, ctx)
            
            if not pick:
                row.append(f"{'N/A':<15}")
                continue
            
            ppl = pick['ppl']
            delta = ((ppl - base_ppl) / base_ppl) * 100
            
            # Kategorisierung
            if delta < 1:
                status = "✓"
            elif delta < 10:
                status = "~"
            elif delta < 100:
                status = "!"
            else:
                status = "✗ BROKEN"
            
            row.append(f"{delta:>6.2f}% {status:<7}")
        
        print(" ".join(row))
    
    print("\n" + "="*80)
    print("Legende: ✓ <1% | ~ <10% | ! <100% | ✗ BROKEN ≥100%")
    print("="*80 + "\n")
    
    # Empfehlung
    print("\nEmpfehlung für Visualisierung:")
    print("─────────────────────────────────")
    print("• Plot 1: Memory vs Context (Mistral) — bleibt wie ist")
    print("• Plot 2: Stability Heatmap — 4 Modelle × 3 Präzisionen, farbcodiert")
    print("• Plot 3: GQA Scatter — INT4/INT2 nur, log-scale, minimale Labels")
    print()

if __name__ == "__main__":
    main()
