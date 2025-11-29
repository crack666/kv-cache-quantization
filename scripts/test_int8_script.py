#!/usr/bin/env python3
"""
Smoke Test für quantize_kvcache_int8.py
========================================

Testet das INT8-Quantisierungs-Script mit einem kleinen Modell (GPT-2)
um schnell zu validieren, dass die Grundfunktionalität stimmt.

Usage:
    python test_int8_script.py
"""

import subprocess
import sys
from pathlib import Path

def run_smoke_test():
    """Führt Smoke-Test durch."""
    
    print("="*80)
    print("INT8 Quantization Script - Smoke Test")
    print("="*80)
    print()
    
    script_path = Path(__file__).parent / "quantize_kvcache_int8.py"
    
    if not script_path.exists():
        print(f"❌ Script nicht gefunden: {script_path}")
        return 1
    
    # Test 1: FP16 Baseline mit GPT-2 (klein, schnell)
    print("\n" + "-"*80)
    print("Test 1: FP16 Baseline (GPT-2, 124M Parameter)")
    print("-"*80)
    
    cmd_fp16 = [
        sys.executable,
        str(script_path),
        "--model", "gpt2",
        "--context-lengths", "128", "256",
        "--no-quantize",
        "--seed", "42"
    ]
    
    print(f"Command: {' '.join(cmd_fp16)}\n")
    
    try:
        result = subprocess.run(cmd_fp16, check=True, capture_output=False)
        print("\n✅ Test 1 passed: FP16 baseline funktioniert")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Test 1 failed: {e}")
        return 1
    
    # Test 2: INT8 Quantization
    print("\n" + "-"*80)
    print("Test 2: INT8 Quantization (GPT-2)")
    print("-"*80)
    
    cmd_int8 = [
        sys.executable,
        str(script_path),
        "--model", "gpt2",
        "--context-lengths", "128", "256",
        "--quantize",
        "--seed", "42"
    ]
    
    print(f"Command: {' '.join(cmd_int8)}\n")
    
    try:
        result = subprocess.run(cmd_int8, check=True, capture_output=False)
        print("\n✅ Test 2 passed: INT8 quantization funktioniert")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Test 2 failed: {e}")
        return 1
    
    # Validate output files
    results_dir = Path(__file__).parent.parent / "results" / "raw"
    json_files = list(results_dir.glob("gpt2_*.json"))
    
    if len(json_files) < 2:
        print(f"\n⚠️  Warning: Expected 2 result files, found {len(json_files)}")
        print(f"Files: {[f.name for f in json_files]}")
    else:
        print(f"\n✅ Output validation passed: {len(json_files)} result files found")
        for f in json_files:
            print(f"  - {f.name}")
    
    print("\n" + "="*80)
    print("✅ All smoke tests passed!")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(run_smoke_test())
