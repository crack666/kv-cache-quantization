#!/usr/bin/env python3
"""
Aggregate and display KV-cache quantization experiment results.

Features:
- Per-model compression ratio calculation (uses each model's FP16 as baseline)
- PPL-delta calculation relative to FP16 baseline
- Grouped output by model for clarity
- LaTeX table export for thesis integration
- JSON summary export for further analysis

Usage:
    python aggregate_results.py --table --latest
    python aggregate_results.py --latex --output results/tables/
    python aggregate_results.py --summary
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple


def load_all_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all JSON result files from the results directory."""
    results = []
    for json_file in results_dir.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                data['_source_file'] = json_file.name
                results.append(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load {json_file}: {e}")
    return results


def get_model_key(result: Dict[str, Any]) -> str:
    """Extract a consistent model key from result data."""
    # Handle both schema v1.0 and v2.0
    if 'config' in result:
        return result['config'].get('model', 'unknown')
    elif 'model' in result:
        return result['model']
    else:
        # Try to infer from filename
        return result.get('_source_file', 'unknown').split('_')[0]


def get_precision(result: Dict[str, Any]) -> str:
    """Extract precision from result data."""
    if 'config' in result:
        # Try multiple field names for backwards compatibility
        config = result['config']
        return config.get('kv_precision') or config.get('kv_cache_precision') or 'fp16'
    elif 'kv_precision' in result:
        return result['kv_precision']
    elif 'kv_cache_precision' in result:
        return result['kv_cache_precision']
    return 'fp16'


def get_backend(result: Dict[str, Any]) -> str:
    """Extract backend from result data."""
    if 'config' in result:
        return result['config'].get('backend', '-')
    elif 'backend' in result:
        return result['backend']
    return '-'


def extract_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from a result entry."""
    metrics = {
        'model': get_model_key(result),
        'precision': get_precision(result),
        'backend': get_backend(result),
        'bytes_per_token': None,
        'ppl': None,
        'timestamp': None,
        'source_file': result.get('_source_file', ''),
    }
    
    # Extract bytes_per_token
    if 'summary' in result and result['summary']:
        metrics['bytes_per_token'] = result['summary'].get('avg_bytes_per_token')
    elif 'kv_bytes_per_token' in result:
        metrics['bytes_per_token'] = result['kv_bytes_per_token']
    
    # Extract PPL (handle both avg_ppl, avg_perplexity, and single ppl)
    if 'summary' in result and result['summary']:
        summary = result['summary']
        metrics['ppl'] = summary.get('avg_ppl') or summary.get('avg_perplexity')
    elif 'perplexity' in result:
        metrics['ppl'] = result['perplexity']
    
    # Extract timestamp
    if 'metadata' in result:
        metrics['timestamp'] = result['metadata'].get('timestamp')
    elif 'timestamp' in result:
        metrics['timestamp'] = result['timestamp']
    
    return metrics


def group_by_model(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group results by model name."""
    grouped = defaultdict(list)
    for r in results:
        metrics = extract_metrics(r)
        if metrics['bytes_per_token'] is not None:  # Only include valid results
            grouped[metrics['model']].append(metrics)
    return dict(grouped)


def find_fp16_baseline(model_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Find the FP16 baseline for a given model's results."""
    for r in model_results:
        if r['precision'].lower() == 'fp16':
            return r
    return None


def calculate_compression_and_delta(
    model_results: List[Dict[str, Any]], 
    fp16_baseline: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Calculate compression ratio and PPL delta relative to FP16 baseline."""
    baseline_bpt = fp16_baseline['bytes_per_token']
    baseline_ppl = fp16_baseline['ppl']
    
    enriched = []
    for r in model_results:
        r = r.copy()  # Don't mutate original
        
        if r['bytes_per_token'] and baseline_bpt:
            r['compression'] = baseline_bpt / r['bytes_per_token']
        else:
            r['compression'] = 1.0
            
        if r['ppl'] and baseline_ppl and baseline_ppl > 0:
            r['ppl_delta_pct'] = ((r['ppl'] - baseline_ppl) / baseline_ppl) * 100
        else:
            r['ppl_delta_pct'] = 0.0
            
        r['is_baseline'] = (r['precision'].lower() == 'fp16')
        enriched.append(r)
    
    return enriched


def get_short_model_name(full_name: str) -> str:
    """Convert full model path to short display name."""
    name_map = {
        'gpt2': 'GPT-2',
        'openai-community/gpt2': 'GPT-2',
        'Qwen/Qwen2-0.5B': 'Qwen2-0.5B',
        'mistralai/Mistral-7B-v0.1': 'Mistral-7B',
    }
    return name_map.get(full_name, full_name.split('/')[-1])


def format_bytes(bytes_val: float) -> str:
    """Format bytes per token with comma separators."""
    if bytes_val is None:
        return '-'
    return f"{int(bytes_val):,}"


def format_compression(ratio: float) -> str:
    """Format compression ratio."""
    return f"{ratio:.1f}×"


def format_ppl(ppl: float) -> str:
    """Format perplexity value."""
    if ppl is None:
        return '-'
    return f"{ppl:.3f}"


def format_ppl_delta(delta_pct: float, is_baseline: bool) -> str:
    """Format PPL delta percentage."""
    if is_baseline:
        return "baseline"
    if delta_pct > 100:
        return f"+{delta_pct:.0f}% ❌"
    elif delta_pct > 1:
        return f"+{delta_pct:.1f}%"
    elif delta_pct < -0.05:  # Only show negative if meaningful
        return f"{delta_pct:.1f}%"
    else:
        return f"+{abs(delta_pct):.1f}%"


def format_markdown_table(grouped_results: Dict[str, List[Dict[str, Any]]]) -> str:
    """Generate a markdown table grouped by model with proper compression ratios."""
    lines = []
    lines.append("# KV-Cache Quantization Results\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    
    # Summary statistics
    total_experiments = sum(len(r) for r in grouped_results.values())
    lines.append(f"Total experiments: {total_experiments} across {len(grouped_results)} models\n")
    
    # Main table header
    lines.append("| Model | Precision | Backend | bytes/token | Compression | PPL | Δ PPL |")
    lines.append("|-------|-----------|---------|-------------|-------------|-----|-------|")
    
    # Sort models for consistent output
    model_order = ['gpt2', 'openai-community/gpt2', 'Qwen/Qwen2-0.5B', 'mistralai/Mistral-7B-v0.1']
    sorted_models = []
    for m in model_order:
        if m in grouped_results:
            sorted_models.append(m)
    # Add any models not in predefined order
    for m in grouped_results:
        if m not in sorted_models:
            sorted_models.append(m)
    
    for model in sorted_models:
        model_results = grouped_results[model]
        fp16_baseline = find_fp16_baseline(model_results)
        
        if fp16_baseline is None:
            # No FP16 baseline, use first result as reference
            if model_results:
                fp16_baseline = model_results[0]
            else:
                continue
        
        # Calculate compression and delta
        enriched = calculate_compression_and_delta(model_results, fp16_baseline)
        
        # Sort: FP16 first, then by precision
        precision_order = {'fp16': 0, 'int8': 1, 'int4': 2, 'int2': 3}
        enriched.sort(key=lambda x: (precision_order.get(x['precision'].lower(), 99), x['backend']))
        
        short_name = get_short_model_name(model)
        first_row = True
        
        for r in enriched:
            display_model = short_name if first_row else ""
            first_row = False
            
            lines.append(
                f"| {display_model} | {r['precision'].upper()} | {r['backend']} | "
                f"{format_bytes(r['bytes_per_token'])} | {format_compression(r['compression'])} | "
                f"{format_ppl(r['ppl'])} | {format_ppl_delta(r['ppl_delta_pct'], r['is_baseline'])} |"
            )
        
        # Add separator between models
        lines.append("|---|---|---|---|---|---|---|")
    
    # Remove last separator
    if lines[-1].startswith("|---|"):
        lines = lines[:-1]
    
    return "\n".join(lines)


def format_latex_table(grouped_results: Dict[str, List[Dict[str, Any]]]) -> str:
    """Generate LaTeX table for thesis integration."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{KV-Cache Quantization Results}")
    lines.append(r"\label{tab:kv-quantization-results}")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Precision} & \textbf{bytes/token} & \textbf{Compression} & \textbf{PPL} & \textbf{$\Delta$ PPL} \\")
    lines.append(r"\midrule")
    
    # Same model ordering as markdown
    model_order = ['gpt2', 'openai-community/gpt2', 'Qwen/Qwen2-0.5B', 'mistralai/Mistral-7B-v0.1']
    sorted_models = []
    for m in model_order:
        if m in grouped_results:
            sorted_models.append(m)
    for m in grouped_results:
        if m not in sorted_models:
            sorted_models.append(m)
    
    for i, model in enumerate(sorted_models):
        model_results = grouped_results[model]
        fp16_baseline = find_fp16_baseline(model_results)
        
        if fp16_baseline is None:
            if model_results:
                fp16_baseline = model_results[0]
            else:
                continue
        
        enriched = calculate_compression_and_delta(model_results, fp16_baseline)
        
        precision_order = {'fp16': 0, 'int8': 1, 'int4': 2, 'int2': 3}
        enriched.sort(key=lambda x: (precision_order.get(x['precision'].lower(), 99), x['backend']))
        
        short_name = get_short_model_name(model)
        
        for j, r in enumerate(enriched):
            if j == 0:
                model_cell = short_name
            else:
                model_cell = ""
            
            # Format delta for LaTeX
            ppl_delta = r.get('ppl_delta_pct') or 0
            if r['is_baseline']:
                delta_str = "---"
            elif ppl_delta > 100:
                delta_str = f"+{ppl_delta:.0f}\\%"
            elif ppl_delta < -0.05:  # Only show negative if meaningful
                delta_str = f"{ppl_delta:.1f}\\%"
            else:
                delta_str = f"+{abs(ppl_delta):.1f}\\%"
            
            bpt = int(r['bytes_per_token']) if r['bytes_per_token'] else 0
            ppl_val = r['ppl'] if r['ppl'] else 0
            compression = r['compression'] if r['compression'] else 1.0
            
            lines.append(
                f"{model_cell} & {r['precision'].upper()} & "
                f"{bpt:,} & {compression:.1f}$\\times$ & "
                f"{ppl_val:.3f} & {delta_str} \\\\"
            )
        
        if i < len(sorted_models) - 1:
            lines.append(r"\midrule")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def print_summary(grouped_results: Dict[str, List[Dict[str, Any]]]):
    """Print a summary of key findings."""
    print("\n" + "=" * 60)
    print("KV-CACHE QUANTIZATION SUMMARY")
    print("=" * 60)
    
    for model in grouped_results:
        model_results = grouped_results[model]
        fp16_baseline = find_fp16_baseline(model_results)
        
        if not fp16_baseline:
            continue
        
        # Skip if essential values are missing
        if fp16_baseline.get('bytes_per_token') is None or fp16_baseline.get('ppl') is None:
            continue
            
        short_name = get_short_model_name(model)
        ppl_val = fp16_baseline['ppl'] if fp16_baseline['ppl'] else 0
        print(f"\n{short_name}:")
        print(f"  FP16 baseline: {format_bytes(fp16_baseline['bytes_per_token'])} bytes/token, PPL={ppl_val:.3f}")
        
        enriched = calculate_compression_and_delta(model_results, fp16_baseline)
        
        for r in enriched:
            if not r['is_baseline']:
                ppl_delta = r.get('ppl_delta_pct', 0) or 0
                status = "✅" if ppl_delta < 5 else "⚠️" if ppl_delta < 50 else "❌"
                print(f"  {r['precision'].upper()} ({r['backend']}): {format_compression(r['compression'])} compression, Δ={ppl_delta:+.1f}% {status}")


def filter_latest_per_config(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only the latest result for each (model, precision, backend) combination."""
    latest = {}
    
    for r in results:
        metrics = extract_metrics(r)
        key = (metrics['model'], metrics['precision'], metrics['backend'])
        
        if key not in latest:
            latest[key] = r
        else:
            # Compare timestamps (handle None values)
            existing_ts = extract_metrics(latest[key]).get('timestamp') or ''
            current_ts = metrics.get('timestamp') or ''
            if current_ts > existing_ts:
                latest[key] = r
    
    return list(latest.values())


def main():
    parser = argparse.ArgumentParser(description="Aggregate KV-cache quantization results")
    parser.add_argument('--results-dir', type=Path, 
                        default=Path(__file__).parent.parent / 'results' / 'raw',
                        help='Directory containing result JSON files')
    parser.add_argument('--table', action='store_true',
                        help='Output markdown table')
    parser.add_argument('--latex', action='store_true',
                        help='Output LaTeX table')
    parser.add_argument('--summary', action='store_true',
                        help='Output summary statistics')
    parser.add_argument('--output', type=Path,
                        help='Output file path (default: stdout)')
    parser.add_argument('--latest', action='store_true',
                        help='Keep only latest result per configuration')
    parser.add_argument('--all', action='store_true',
                        help='Show all output formats')
    
    args = parser.parse_args()
    
    # Default to table if no output specified
    if not (args.table or args.latex or args.summary or args.all):
        args.table = True
    
    # Load results
    results = load_all_results(args.results_dir)
    
    if not results:
        print(f"No results found in {args.results_dir}")
        return
    
    # Filter to latest if requested
    if args.latest:
        results = filter_latest_per_config(results)
    
    # Group by model
    grouped = group_by_model(results)
    
    output_parts = []
    
    if args.table or args.all:
        output_parts.append(format_markdown_table(grouped))
    
    if args.latex or args.all:
        output_parts.append("\n\n% LaTeX Table\n")
        output_parts.append(format_latex_table(grouped))
    
    if args.summary or args.all:
        print_summary(grouped)
    
    # Output
    output_text = "\n".join(output_parts)
    
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            f.write(output_text)
        print(f"Output written to {args.output}")
    else:
        print(output_text)


if __name__ == '__main__':
    main()
