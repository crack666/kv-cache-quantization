#!/usr/bin/env python3
"""
Model Architecture Analyzer

Analysiert die interne Architektur von Hugging Face Modellen, 
insbesondere im Hinblick auf KV-Cache Quantisierbarkeit.

Dieses Skript extrahiert:
- Attention-Architektur (MHA/GQA/MQA)
- KV-Heads und Query-Heads Verhältnis
- Theoretische KV-Cache Größe pro Token
- Quantisierbarkeits-Einschätzung basierend auf GQA-Ratio

Verwendung:
    python analyze_model_architecture.py mistralai/Mistral-7B-v0.1
    python analyze_model_architecture.py Qwen/Qwen2-7B Qwen/Qwen2-0.5B gpt2

Wissenschaftlicher Hintergrund:
    Die GQA-Ratio (Grouped Query Attention) beeinflusst die Quantisierbarkeit:
    - Höhere GQA-Ratio = weniger KV-Heads = weniger Redundanz
    - Quantisierungsfehler in einem KV-Head propagieren auf alle abhängigen Query-Heads
    - Empirisch beobachtet: GQA 7:1 (Qwen) bricht bei INT4, GQA 4:1 (Mistral) funktioniert
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class ModelArchitecture:
    """Speichert die Architektur-Details eines Modells."""
    model_name: str
    num_layers: int
    num_attention_heads: int  # Query heads
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    gqa_ratio: int
    attention_type: str  # MHA, GQA, MQA
    kv_cache_bytes_per_token_fp16: int
    quantizability_score: str  # ✅ Gut, ⚠️ Kritisch, ❌ Problematisch


def analyze_model(model_name: str) -> ModelArchitecture:
    """
    Analysiert die Architektur eines Hugging Face Modells.
    
    Args:
        model_name: Hugging Face Model ID (z.B. 'mistralai/Mistral-7B-v0.1')
    
    Returns:
        ModelArchitecture mit allen relevanten Details
    """
    from transformers import AutoConfig
    
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    # Extrahiere Architektur-Parameter
    num_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    
    # KV-Heads: Falls nicht explizit definiert, entspricht es num_attention_heads (MHA)
    num_kv_heads = getattr(config, 'num_key_value_heads', num_attention_heads)
    
    # Head Dimension
    head_dim = hidden_size // num_attention_heads
    
    # GQA Ratio berechnen
    gqa_ratio = num_attention_heads // num_kv_heads
    
    # Attention-Typ bestimmen
    if gqa_ratio == 1:
        attention_type = "MHA (Multi-Head Attention)"
    elif num_kv_heads == 1:
        attention_type = "MQA (Multi-Query Attention)"
    else:
        attention_type = f"GQA (Grouped Query Attention)"
    
    # KV-Cache Größe pro Token in FP16 (bytes)
    # Formel: num_layers × 2 (K+V) × num_kv_heads × head_dim × 2 (bytes für FP16)
    kv_cache_bytes_per_token = num_layers * 2 * num_kv_heads * head_dim * 2
    
    # Quantisierbarkeits-Einschätzung basierend auf GQA-Ratio
    if gqa_ratio <= 2:
        quantizability = "✅ Sehr gut (hohe Redundanz)"
    elif gqa_ratio <= 4:
        quantizability = "✅ Gut (moderate Redundanz)"
    elif gqa_ratio <= 6:
        quantizability = "⚠️ Kritisch (wenig Redundanz)"
    else:
        quantizability = "❌ Problematisch (minimale Redundanz)"
    
    return ModelArchitecture(
        model_name=model_name,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        gqa_ratio=gqa_ratio,
        attention_type=attention_type,
        kv_cache_bytes_per_token_fp16=kv_cache_bytes_per_token,
        quantizability_score=quantizability
    )


def print_analysis(arch: ModelArchitecture, verbose: bool = True):
    """Gibt die Architektur-Analyse formatiert aus."""
    
    print(f"\n{'='*70}")
    print(f"MODELL: {arch.model_name}")
    print('='*70)
    
    print(f"\n📐 Architektur:")
    print(f"   Layers:           {arch.num_layers}")
    print(f"   Hidden Size:      {arch.hidden_size}")
    print(f"   Head Dimension:   {arch.head_dim}")
    print(f"   Query Heads:      {arch.num_attention_heads}")
    print(f"   KV Heads:         {arch.num_kv_heads}")
    print(f"   Attention Type:   {arch.attention_type}")
    print(f"   GQA Ratio:        {arch.gqa_ratio}:1")
    
    print(f"\n💾 KV-Cache (FP16):")
    print(f"   Bytes/Token:      {arch.kv_cache_bytes_per_token_fp16:,} bytes")
    print(f"   KB/1K Tokens:     {arch.kv_cache_bytes_per_token_fp16 * 1024 / 1024:.1f} KB")
    print(f"   MB/16K Tokens:    {arch.kv_cache_bytes_per_token_fp16 * 16384 / 1024 / 1024:.1f} MB")
    
    print(f"\n🎯 Quantisierbarkeits-Einschätzung:")
    print(f"   {arch.quantizability_score}")
    
    if verbose and arch.gqa_ratio > 4:
        print(f"\n⚠️  WARNUNG: Hohes GQA-Ratio ({arch.gqa_ratio}:1)")
        print(f"   → Jeder KV-Head bedient {arch.gqa_ratio} Query-Heads")
        print(f"   → Quantisierungsfehler werden {arch.gqa_ratio}× verstärkt")
        print(f"   → INT4/INT2 KV-Cache Quantisierung kann fehlschlagen!")


def compare_models(models: list[str]) -> dict:
    """
    Vergleicht mehrere Modelle und gibt eine Zusammenfassung zurück.
    """
    results = []
    
    for model_name in models:
        try:
            arch = analyze_model(model_name)
            results.append(arch)
            print_analysis(arch)
        except Exception as e:
            print(f"\n❌ Fehler bei {model_name}: {e}")
    
    if len(results) > 1:
        print(f"\n{'='*70}")
        print("VERGLEICHSTABELLE")
        print('='*70)
        
        print(f"\n{'Modell':<25} {'Layers':>7} {'Q-Heads':>8} {'KV-Heads':>9} {'GQA':>5} {'Quant.':<20}")
        print("-" * 80)
        
        for arch in sorted(results, key=lambda x: x.gqa_ratio):
            # Kurzer Name
            short_name = arch.model_name.split('/')[-1][:24]
            quant_short = arch.quantizability_score.split('(')[0].strip()
            print(f"{short_name:<25} {arch.num_layers:>7} {arch.num_attention_heads:>8} {arch.num_kv_heads:>9} {arch.gqa_ratio:>4}:1 {quant_short:<20}")
    
    return {
        'timestamp': datetime.now().isoformat(),
        'models': [
            {
                'model_name': a.model_name,
                'num_layers': a.num_layers,
                'num_attention_heads': a.num_attention_heads,
                'num_kv_heads': a.num_kv_heads,
                'head_dim': a.head_dim,
                'hidden_size': a.hidden_size,
                'gqa_ratio': a.gqa_ratio,
                'attention_type': a.attention_type,
                'kv_cache_bytes_per_token_fp16': a.kv_cache_bytes_per_token_fp16,
                'quantizability_score': a.quantizability_score,
            }
            for a in results
        ]
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analysiert Modell-Architekturen für KV-Cache Quantisierung",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
    python analyze_model_architecture.py mistralai/Mistral-7B-v0.1
    python analyze_model_architecture.py Qwen/Qwen2-7B Qwen/Qwen2-0.5B gpt2
    python analyze_model_architecture.py --output arch.json meta-llama/Llama-2-7b-hf

Wissenschaftlicher Hintergrund:
    Das GQA-Ratio (Grouped Query Attention) ist ein kritischer Faktor für die 
    Quantisierbarkeit des KV-Cache:
    
    - GQA 1:1 (MHA): Maximale Redundanz, INT2 funktioniert oft
    - GQA 4:1: Gute Redundanz, INT4 funktioniert (z.B. Mistral-7B)
    - GQA 7:1+: Kritisch, INT4 kann katastrophal versagen (z.B. Qwen2-7B)
    
    Unsere Experimente zeigen:
    - Mistral-7B (GQA 4:1): INT2-HQQ mit nur +0.4% PPL bei 8× Kompression
    - Qwen2-7B (GQA 7:1): INT4-HQQ mit +2,805,291% PPL (katastrophal)
        """
    )
    
    parser.add_argument('models', nargs='+', 
                       help='Hugging Face Model IDs zum Analysieren')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Optional: Speichere Ergebnisse als JSON')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Weniger Ausgabe')
    
    args = parser.parse_args()
    
    print("🔍 Model Architecture Analyzer für KV-Cache Quantisierung")
    print("=" * 70)
    
    results = compare_models(args.models)
    
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Ergebnisse gespeichert: {output_path}")
    
    print("\n✅ Analyse abgeschlossen")


if __name__ == "__main__":
    main()
