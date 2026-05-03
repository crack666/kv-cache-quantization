#!/usr/bin/env python3
"""
Generate presentation graphics for Pecha Kucha slides.
Outputs to kv-cache-quantization/results/figures/presentation/
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "figures" / "presentation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Style settings for presentation (dark background, large fonts)
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#1a1a2e',
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'axes.edgecolor': 'white',
})


def create_bitwidth_comparison():
    """
    Slide 6: Balkendiagramm FP16 → INT8 → INT4 → INT2
    Zeigt Bitbreite und relative Speichergröße
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    configs = ['FP16\n(Baseline)', 'INT8', 'INT4', 'INT2']
    bits = [16, 8, 4, 2]
    percentages = [100, 50, 25, 12.5]
    
    # Color gradient from red (large) to green (small)
    colors = ['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff']
    
    bars = ax.barh(configs, bits, color=colors, height=0.6, edgecolor='white', linewidth=2)
    
    # Add percentage labels
    for bar, pct, bit in zip(bars, percentages, bits):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{bit} Bit → {pct:.1f}% Speicher',
                va='center', ha='left', fontsize=14, color='white', fontweight='bold')
    
    ax.set_xlabel('Bits pro Wert', fontsize=16)
    ax.set_xlim(0, 24)
    ax.set_title('Quantisierungsstufen: Weniger Bits = Weniger Speicher', 
                 fontsize=20, fontweight='bold', pad=20)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save
    output_path = OUTPUT_DIR / "slide06_bitwidth_comparison.pdf"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_gqa_mechanism():
    """
    Slide 7: GQA-Mechanismus Diagramm
    Zeigt wie 4 Query-Heads sich einen KV-Head teilen
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    query_color = '#6bcb77'  # Green
    kv_color = '#ff6b6b'     # Red
    arrow_color = '#ffd93d'  # Yellow
    
    # Draw Query Heads (4 boxes on the left)
    query_labels = ['Query Head 1', 'Query Head 2', 'Query Head 3', 'Query Head 4']
    query_y_positions = [8, 6, 4, 2]
    
    for i, (label, y) in enumerate(zip(query_labels, query_y_positions)):
        # Query box
        rect = mpatches.FancyBboxPatch((0.5, y - 0.5), 3.5, 1.2,
                                        boxstyle="round,pad=0.05",
                                        facecolor=query_color, edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        ax.text(2.25, y + 0.1, label, ha='center', va='center', 
                fontsize=13, fontweight='bold', color='#1a1a2e')
        
        # Arrow to KV-Head
        ax.annotate('', xy=(8, 5), xytext=(4.2, y + 0.1),
                    arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2.5))
    
    # Draw KV-Head (1 box on the right)
    kv_rect = mpatches.FancyBboxPatch((8, 3.5), 4.5, 3,
                                       boxstyle="round,pad=0.1",
                                       facecolor=kv_color, edgecolor='white', linewidth=3)
    ax.add_patch(kv_rect)
    ax.text(10.25, 5, 'KV-Head 1\n(Key + Value)', ha='center', va='center',
            fontsize=15, fontweight='bold', color='white')
    
    # Title and explanation
    ax.text(7, 9.5, 'Grouped Query Attention (4:1)', 
            ha='center', va='center', fontsize=22, fontweight='bold', color='white')
    
    ax.text(7, 0.5, '→ Quantisierungsfehler im KV-Head beeinflusst ALLE 4 Query-Heads',
            ha='center', va='center', fontsize=14, color='#ffd93d', fontweight='bold')
    
    # Ratio label
    ax.text(5.5, 5, '4:1', ha='center', va='center', 
            fontsize=28, fontweight='bold', color='white',
            bbox=dict(boxstyle='circle,pad=0.3', facecolor='#4d96ff', edgecolor='white', linewidth=2))
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / "slide07_gqa_mechanism.pdf"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_expectation_vs_reality():
    """
    Slide 13: Erwartung vs. Realität
    Zeigt den Kontrast zwischen Hypothese und tatsächlichem Ergebnis
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    models = ['Qwen2-7B\n(7:1)', 'Yi-1.5-9B\n(8:1)']
    
    # LEFT: Expectation (what we thought would happen)
    expected = ['BROKEN ✗', 'WORSE ✗✗']
    expected_colors = ['#ff6b6b', '#cc4444']
    
    ax1.barh(models, [1, 1], color=expected_colors, height=0.5, edgecolor='white', linewidth=2)
    ax1.set_xlim(0, 1.5)
    ax1.set_title('ERWARTUNG', fontsize=20, fontweight='bold', color='#ff6b6b', pad=15)
    
    for i, (model, exp) in enumerate(zip(models, expected)):
        ax1.text(0.5, i, exp, ha='center', va='center', 
                 fontsize=18, fontweight='bold', color='white')
    
    ax1.set_xticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    
    # RIGHT: Reality (what actually happened)
    reality = ['BROKEN ✗', 'WORKS ✓']
    reality_colors = ['#ff6b6b', '#6bcb77']
    
    ax2.barh(models, [1, 1], color=reality_colors, height=0.5, edgecolor='white', linewidth=2)
    ax2.set_xlim(0, 1.5)
    ax2.set_title('REALITÄT', fontsize=20, fontweight='bold', color='#6bcb77', pad=15)
    
    for i, (model, real) in enumerate(zip(models, reality)):
        ax2.text(0.5, i, real, ha='center', va='center',
                 fontsize=18, fontweight='bold', color='white' if 'BROKEN' in real else '#1a1a2e')
    
    ax2.set_xticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    
    # Add surprise annotation
    fig.text(0.5, 0.02, '⚡ Yi (8:1) ist ROBUSTER als erwartet – GQA-Ratio ist kein Prädiktor!',
             ha='center', fontsize=16, fontweight='bold', color='#ffd93d')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
    output_path = OUTPUT_DIR / "slide13_expectation_vs_reality.pdf"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_qwen2_explosion():
    """
    Slide 14: Qwen2-7B PPL Explosion
    Dramatische Darstellung des katastrophalen Versagens
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    configs = ['FP16', 'INT8', 'INT4', 'INT2']
    # Using log scale for visualization, actual values shown in labels
    ppl_values = [1.11, 1.11, 264956, 1400000]
    ppl_log = [np.log10(v) for v in ppl_values]
    
    colors = ['#6bcb77', '#6bcb77', '#ff6b6b', '#cc4444']
    
    bars = ax.bar(configs, ppl_log, color=colors, edgecolor='white', linewidth=2, width=0.6)
    
    # Add actual PPL values as labels
    for bar, ppl in zip(bars, ppl_values):
        if ppl > 100:
            label = f'{ppl:,.0f}'
            y_offset = 0.2
        else:
            label = f'{ppl:.2f}'
            y_offset = 0.1
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_offset,
                label, ha='center', va='bottom', fontsize=14, fontweight='bold', color='white')
    
    ax.set_ylabel('Perplexity (log₁₀)', fontsize=14)
    ax.set_title('Qwen2-7B: Katastrophales Versagen bei INT4', 
                 fontsize=20, fontweight='bold', color='#ff6b6b', pad=20)
    
    # Add "Faktor 240.000!" annotation
    ax.annotate('Faktor\n240.000!', xy=(2, ppl_log[2]), xytext=(2.7, 4),
                fontsize=16, fontweight='bold', color='#ffd93d',
                arrowprops=dict(arrowstyle='->', color='#ffd93d', lw=2))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 7)
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / "slide14_qwen2_explosion.pdf"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_practical_recommendations():
    """
    Slide 16: Ampel-Empfehlungen
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    recommendations = [
        ('INT8', '🟢', '#6bcb77', 'Universell sicher (<1% PPL)', 8),
        ('INT4', '🟡', '#ffd93d', 'Modellspezifisch validieren', 5),
        ('INT2', '🔴', '#ff6b6b', 'Nur für robuste Modelle', 2),
    ]
    
    for label, emoji, color, desc, y in recommendations:
        # Traffic light circle
        circle = plt.Circle((1.5, y), 0.8, color=color, ec='white', linewidth=3)
        ax.add_patch(circle)
        
        # Label
        ax.text(3, y, label, fontsize=24, fontweight='bold', color='white', va='center')
        
        # Description
        ax.text(5, y, desc, fontsize=16, color='white', va='center')
    
    ax.set_title('Praktische Empfehlungen', fontsize=22, fontweight='bold', color='white', pad=20)
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / "slide16_recommendations.pdf"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"✓ Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("Generating presentation graphics...")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    create_bitwidth_comparison()
    create_gqa_mechanism()
    create_expectation_vs_reality()
    create_qwen2_explosion()
    create_practical_recommendations()
    
    print(f"\n✓ All graphics saved to: {OUTPUT_DIR}")
    print("  Use the PNG files for PowerPoint, PDF for higher quality.")
