#!/usr/bin/env python3
"""
Aggregate profiling results (profile_*.json) into thesis-ready tables.

Creates comprehensive CSV/LaTeX tables with all measured metrics across
models, precisions, and context lengths for thesis appendix.

Usage:
    python aggregate_profiling_results.py                    # Generate all formats
    python aggregate_profiling_results.py --csv-only         # CSV only
    python aggregate_profiling_results.py --latex-only       # LaTeX only
    python aggregate_profiling_results.py --context 4096     # Filter to specific context
"""

import json
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any


def load_profiling_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all profile_*.json files."""
    results = []
    for json_file in results_dir.glob("profile_*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                results.append(data)
                print(f"✓ Loaded {json_file.name}")
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠ Warning: Could not load {json_file}: {e}")
    return results


def extract_measurements_to_dataframe(profiling_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert profiling JSON data to flat DataFrame."""
    rows = []
    
    for profile in profiling_data:
        model = profile.get('model', 'unknown')
        model_short = model.split('/')[-1]  # e.g., "Mistral-7B-v0.1"
        timestamp = profile.get('timestamp', '')
        
        for measurement in profile.get('measurements', []):
            row = {
                'Model': model_short,
                'Config': measurement.get('config', ''),
                'Context': measurement.get('context_len', 0),
                'Tokens Generated': measurement.get('tokens_generated', 0),
                'Time (ms)': measurement.get('total_ms', 0),
                'Throughput (tok/s)': measurement.get('tokens_per_sec', 0),
                'KV Cache (MB)': measurement.get('kv_cache_mb', 0),
                'Perplexity': measurement.get('perplexity', 0),
                'Quant (ms)': measurement.get('quant_ms', 0),
                'Dequant (ms)': measurement.get('dequant_ms', 0),
                'Overhead (%)': measurement.get('overhead_pct', 0),
                'Power (W)': measurement.get('avg_watts', 0),
                'Energy (mJ/tok)': measurement.get('energy_mj_per_token', 0),
                'Timestamp': timestamp
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort for readability: Model, Config, Context
    df = df.sort_values(['Model', 'Config', 'Context'])
    
    return df


def calculate_delta_ppl(df: pd.DataFrame) -> pd.DataFrame:
    """Add ΔPPL column (relative to FP16 baseline per model+context)."""
    df = df.copy()
    df['ΔPPL (%)'] = 0.0
    
    for model in df['Model'].unique():
        for context in df['Context'].unique():
            mask = (df['Model'] == model) & (df['Context'] == context)
            fp16_mask = mask & (df['Config'] == 'FP16')
            
            if fp16_mask.any():
                fp16_ppl = df.loc[fp16_mask, 'Perplexity'].iloc[0]
                
                for idx in df[mask].index:
                    ppl = df.loc[idx, 'Perplexity']
                    if fp16_ppl > 0:
                        df.loc[idx, 'ΔPPL (%)'] = ((ppl / fp16_ppl) - 1) * 100
    
    return df


def save_csv(df: pd.DataFrame, output_dir: Path, context_filter: int = None):
    """Save DataFrame as CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if context_filter:
        df_filtered = df[df['Context'] == context_filter]
        output_file = output_dir / f"profiling_results_ctx{context_filter}.csv"
    else:
        df_filtered = df
        output_file = output_dir / "profiling_results_all.csv"
    
    df_filtered.to_csv(output_file, index=False, float_format='%.2f')
    print(f"✅ CSV saved: {output_file}")


def save_latex_table(df: pd.DataFrame, output_dir: Path, context_filter: int = None):
    """Generate LaTeX table for thesis appendix."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if context_filter:
        df_filtered = df[df['Context'] == context_filter]
        output_file = output_dir / f"profiling_table_ctx{context_filter}.tex"
        caption = f"KV-Cache Profiling Results at {context_filter} Tokens Context"
    else:
        df_filtered = df
        output_file = output_dir / "profiling_table_all.tex"
        caption = "Complete KV-Cache Profiling Results (All Contexts)"
    
    # Select key columns for thesis table
    columns_to_export = [
        'Model', 'Config', 'Context', 'Throughput (tok/s)', 
        'KV Cache (MB)', 'Perplexity', 'ΔPPL (%)', 'Overhead (%)'
    ]
    
    df_export = df_filtered[columns_to_export].copy()

    # Sort for readability: within each model and context, compare configs directly.
    # Desired order: Model -> Context -> Config (FP16, INT8, INT4, INT2)
    config_order = ['FP16', 'INT8 (HQQ)', 'INT4 (HQQ)', 'INT2 (HQQ)']
    df_export['Config'] = pd.Categorical(df_export['Config'], categories=config_order, ordered=True)
    df_export = df_export.sort_values(['Model', 'Context', 'Config'], kind='mergesort')
    
    # Rename column for LaTeX compatibility (replace Unicode Δ with \Delta command)
    df_export.rename(columns={'ΔPPL (%)': r'$\Delta$PPL (\%)', 'Overhead (%)': r'Overhead (\%)'}, inplace=True)
    
    # Format columns for LaTeX
    df_export['Throughput (tok/s)'] = df_export['Throughput (tok/s)'].apply(lambda x: f"{x:.1f}")
    df_export['KV Cache (MB)'] = df_export['KV Cache (MB)'].apply(lambda x: f"{x:.1f}")
    df_export['Perplexity'] = df_export['Perplexity'].apply(lambda x: f"{x:.2f}")
    df_export[r'$\Delta$PPL (\%)'] = df_export[r'$\Delta$PPL (\%)'].apply(lambda x: f"{x:+.1f}" if abs(x) > 0.01 else "0.0")
    df_export[r'Overhead (\%)'] = df_export[r'Overhead (\%)'].apply(lambda x: f"{x:.1f}")
    
    # Generate LaTeX with longtable for multi-page support in landscape
    # Use p{width} columns for text fields and right-aligned p{} for last two cols
    latex = df_export.to_latex(
        index=False,
        column_format=r'|p{2.2cm}|p{2.5cm}|r|r|r|r|>{\raggedleft\arraybackslash}p{1.3cm}|>{\raggedleft\arraybackslash}p{1.3cm}|',
        escape=False,
        longtable=True,  # Use longtable for landscape compatibility
        caption='Vollständige KV-Cache Profiling-Ergebnisse (alle Kontexte)',
        label='tab:profiling_all'
    )

    # Pandas emits a \midrule directly before \endfirsthead/\endhead.
    # With the BHT template this can trigger 'Misplaced \cr' / alignment errors.
    # Dropping those specific midrules keeps the longtable valid and compiles reliably.
    latex = latex.replace('\\midrule\n\\endfirsthead\n', '\\endfirsthead\n')
    latex = latex.replace('\\midrule\n\\endhead\n', '\\endhead\n')
    
    # Remove the continuation caption to keep all pages consistent
    latex = latex.replace('\\caption[]{Vollständige KV-Cache Profiling-Ergebnisse (alle Kontexte)} \\\\\n', '')
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(latex)
    
    print(f"✅ LaTeX table saved: {output_file}")


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total measurements: {len(df)}")
    print(f"Models tested: {df['Model'].nunique()}")
    print(f"Unique configs: {', '.join(df['Config'].unique())}")
    print(f"Context lengths: {', '.join(map(str, sorted(df['Context'].unique())))}")
    
    print("\n--- Average KV-Cache Size Reduction ---")
    for config in ['INT8 (HQQ)', 'INT4 (HQQ)', 'INT2 (HQQ)']:
        if config in df['Config'].values:
            avg_reduction = 100 - (df[df['Config'] == config]['KV Cache (MB)'].mean() / 
                                   df[df['Config'] == 'FP16']['KV Cache (MB)'].mean() * 100)
            print(f"{config:15s}: {avg_reduction:5.1f}% reduction vs FP16")
    
    print("\n--- Quality Impact (Average ΔPPL) ---")
    for config in ['INT8 (HQQ)', 'INT4 (HQQ)', 'INT2 (HQQ)']:
        if config in df['Config'].values:
            avg_delta = df[df['Config'] == config]['ΔPPL (%)'].mean()
            print(f"{config:15s}: {avg_delta:+6.1f}% average ΔPPL")
    
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate profiling results for thesis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv-only", action="store_true", help="Generate CSV only")
    parser.add_argument("--latex-only", action="store_true", help="Generate LaTeX only")
    parser.add_argument("--context", type=int, help="Filter to specific context length")
    parser.add_argument("--output-dir", type=str, default="results/tables",
                        help="Output directory (default: results/tables)")
    
    args = parser.parse_args()
    
    # Load data
    results_dir = Path(__file__).parent.parent / "results" / "raw"
    print(f"Loading profiling results from: {results_dir}\n")
    profiling_data = load_profiling_results(results_dir)
    
    if not profiling_data:
        print("❌ No profiling data found!")
        exit(1)
    
    # Convert to DataFrame
    df = extract_measurements_to_dataframe(profiling_data)
    df = calculate_delta_ppl(df)
    
    # Print summary
    print_summary_statistics(df)
    
    # Save outputs
    output_dir = Path(__file__).parent.parent / args.output_dir
    
    if args.csv_only:
        save_csv(df, output_dir, args.context)
    elif args.latex_only:
        save_latex_table(df, output_dir, args.context)
    else:
        # Generate both
        save_csv(df, output_dir, args.context)
        save_latex_table(df, output_dir, args.context)
    
    print("\n✅ Aggregation complete!")
