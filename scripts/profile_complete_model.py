#!/usr/bin/env python3
"""
Profile complete model: FP16 + INT8 + INT4 + INT2 in one consolidated JSON.

Usage:
    python profile_complete_model.py --model "mistralai/Mistral-7B-v0.1" --contexts 128 256 512 1024 2048 4096
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import subprocess

def run_single_config(model, contexts, config_name, quantize_args, script_dir):
    """Run quantize_kvcache_hf.py for one config and return measurements."""
    cmd = [
        sys.executable,
        str(script_dir / "quantize_kvcache_hf.py"),
        "--model", model,
        "--context-lengths", *[str(c) for c in contexts],
        "--seed", "42",
        "--output-dir", str(script_dir.parent / "results" / "raw" / "temp")
    ]
    cmd.extend(quantize_args)
    
    print(f"\n{'='*60}")
    print(f"Running {config_name}...")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to run {config_name}")
    
    # Find the generated JSON
    temp_dir = script_dir.parent / "results" / "raw" / "temp"
    json_files = list(temp_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No output JSON found for {config_name}")
    
    # Load and return measurements
    latest = max(json_files, key=lambda p: p.stat().st_mtime)
    with open(latest, 'r') as f:
        data = json.load(f)
    
    # Cleanup temp file
    latest.unlink()
    
    return data['measurements']


def main():
    parser = argparse.ArgumentParser(description="Complete model profiling (all configs)")
    parser.add_argument('--model', required=True, help="Model name or path")
    parser.add_argument('--contexts', type=int, nargs='+', required=True,
                       help="Context lengths (e.g., 128 256 512 1024 2048 4096)")
    parser.add_argument('--output-dir', default='results/raw',
                       help="Output directory (default: results/raw)")
    
    args = parser.parse_args()
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    # Create temp dir
    temp_dir = script_dir.parent / "results" / "raw" / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Run all 4 configs
    configs = [
        ("FP16", ["--no-quantize"]),
        ("INT8 (HQQ)", ["--quantize", "--nbits", "8", "--backend", "hqq"]),
        ("INT4 (HQQ)", ["--quantize", "--nbits", "4", "--backend", "hqq"]),
        ("INT2 (HQQ)", ["--quantize", "--nbits", "2", "--backend", "hqq"]),
    ]
    
    all_measurements = []
    
    for config_name, quant_args in configs:
        measurements = run_single_config(args.model, args.contexts, config_name, quant_args, script_dir)
        all_measurements.extend(measurements)
        print(f"✓ {config_name}: {len(measurements)} measurements collected")
    
    # Build consolidated JSON
    model_short = args.model.split('/')[-1].lower().replace('-', '_').replace('.', '_')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    consolidated = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "prompt_tokens": 9,
        "target_contexts": sorted(args.contexts),
        "configs": [c[0] for c in configs],
        "measurements": all_measurements
    }
    
    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"profile_{model_short}_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(consolidated, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ Complete profiling saved to: {output_file.name}")
    print(f"   Model: {consolidated['model']}")
    print(f"   Configs: {len(consolidated['configs'])}")
    print(f"   Contexts: {len(consolidated['target_contexts'])}")
    print(f"   Total measurements: {len(consolidated['measurements'])}")
    print('='*60)
    
    # Cleanup temp dir
    import shutil
    temp_cleanup = script_dir.parent / "results" / "raw" / "temp"
    shutil.rmtree(temp_cleanup, ignore_errors=True)


if __name__ == '__main__':
    main()
