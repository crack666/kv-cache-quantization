#!/usr/bin/env python3
"""
Validierungs-Script: Mehrere Runs um Throughput-Messung zu validieren.
Speichert alle Runs in separaten JSON-Dateien und berechnet Statistiken.
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
TEST_SCRIPT = SCRIPT_DIR / "test_throughput_mistral.py"
RESULTS_DIR = SCRIPT_DIR / "validation_runs"

NUM_RUNS = 3

def run_test(run_id: int) -> dict:
    """Führt einen Test-Run aus und gibt die Ergebnisse zurück."""
    print(f"\n{'='*70}")
    print(f"RUN {run_id}/{NUM_RUNS}")
    print(f"{'='*70}\n")
    
    # Run the test
    result = subprocess.run(
        ["python", str(TEST_SCRIPT)],
        capture_output=False,
        text=True,
        cwd=str(SCRIPT_DIR)
    )
    
    # Read the results
    results_file = SCRIPT_DIR / "throughput_mistral_test.json"
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
        
        # Copy to validation folder with run ID
        RESULTS_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = RESULTS_DIR / f"run_{run_id}_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✅ Run {run_id} saved to: {output_file.name}")
        return data
    else:
        print(f"❌ No results file found for run {run_id}")
        return None


def analyze_runs():
    """Analysiert alle Runs und zeigt Statistiken."""
    RESULTS_DIR.mkdir(exist_ok=True)
    run_files = sorted(RESULTS_DIR.glob("run_*.json"))
    
    if not run_files:
        print("Keine Runs gefunden!")
        return
    
    print(f"\n{'='*80}")
    print(f"VALIDATION ANALYSIS - {len(run_files)} Runs")
    print(f"{'='*80}\n")
    
    # Collect data per config
    configs = {}
    
    for run_file in run_files:
        with open(run_file) as f:
            data = json.load(f)
        
        for measurement in data.get("throughput_measurements", []):
            label = measurement["label"]
            if label not in configs:
                configs[label] = {
                    "tokens_per_second": [],
                    "tpt_ms": [],
                    "ttft_ms": [],
                    "peak_memory_mb": [],
                    "avg_power_watts": [],
                    "energy_per_token_mj": []
                }
            
            configs[label]["tokens_per_second"].append(measurement.get("tokens_per_second", 0))
            configs[label]["tpt_ms"].append(measurement.get("tpt_ms", 0))
            configs[label]["ttft_ms"].append(measurement.get("ttft_ms", 0))
            configs[label]["peak_memory_mb"].append(measurement.get("peak_memory_mb", 0))
            configs[label]["avg_power_watts"].append(measurement.get("avg_power_watts", 0))
            configs[label]["energy_per_token_mj"].append(measurement.get("energy_per_token_mj", 0))
    
    # Calculate statistics
    def stats(values):
        if not values:
            return {"mean": 0, "min": 0, "max": 0, "std": 0}
        n = len(values)
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n if n > 1 else 0
        std = variance ** 0.5
        return {"mean": mean, "min": min(values), "max": max(values), "std": std, "n": n}
    
    # Print results
    print(f"{'Config':<18} {'Tok/s (mean±std)':>20} {'TPT ms':>15} {'Runs':>6}")
    print("-" * 80)
    
    baseline_mean = None
    for label in ["FP16 Baseline", "INT8 (HQQ)", "INT4 (HQQ)", "INT2 (HQQ)"]:
        if label not in configs:
            continue
        
        tps = stats(configs[label]["tokens_per_second"])
        tpt = stats(configs[label]["tpt_ms"])
        
        if baseline_mean is None:
            baseline_mean = tps["mean"]
            ratio = ""
        else:
            ratio = f" ({tps['mean']/baseline_mean:.0%})" if baseline_mean > 0 else ""
        
        print(f"{label:<18} {tps['mean']:>8.1f} ± {tps['std']:>4.1f}{ratio:>6} {tpt['mean']:>10.1f} ± {tpt['std']:>3.1f} {tps['n']:>6}")
    
    print("-" * 80)
    
    # Detailed per-run breakdown
    print(f"\n{'='*80}")
    print("PER-RUN BREAKDOWN (Tokens/s)")
    print(f"{'='*80}")
    print(f"{'Run':<10} {'FP16':>10} {'INT8':>10} {'INT4':>10} {'INT2':>10} {'INT8/FP16':>12}")
    print("-" * 80)
    
    for i, run_file in enumerate(run_files, 1):
        with open(run_file) as f:
            data = json.load(f)
        
        run_data = {}
        for m in data.get("throughput_measurements", []):
            run_data[m["label"]] = m["tokens_per_second"]
        
        fp16 = run_data.get("FP16 Baseline", 0)
        int8 = run_data.get("INT8 (HQQ)", 0)
        int4 = run_data.get("INT4 (HQQ)", 0)
        int2 = run_data.get("INT2 (HQQ)", 0)
        ratio = f"{int8/fp16:.0%}" if fp16 > 0 else "-"
        
        print(f"Run {i:<6} {fp16:>10.1f} {int8:>10.1f} {int4:>10.1f} {int2:>10.1f} {ratio:>12}")
    
    print("-" * 80)
    
    # Conclusion
    if "FP16 Baseline" in configs and "INT8 (HQQ)" in configs:
        fp16_stats = stats(configs["FP16 Baseline"]["tokens_per_second"])
        int8_stats = stats(configs["INT8 (HQQ)"]["tokens_per_second"])
        
        mean_ratio = int8_stats["mean"] / fp16_stats["mean"] if fp16_stats["mean"] > 0 else 0
        
        print(f"\n📊 FAZIT:")
        print(f"   FP16 Baseline: {fp16_stats['mean']:.1f} ± {fp16_stats['std']:.1f} tok/s")
        print(f"   INT8 (HQQ):    {int8_stats['mean']:.1f} ± {int8_stats['std']:.1f} tok/s")
        print(f"   Ratio:         {mean_ratio:.0%} {'✅ INT8 schneller!' if mean_ratio > 1.0 else '⚠️ FP16 schneller'}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        # Only analyze existing runs
        analyze_runs()
    else:
        # Run tests and analyze
        print(f"Starting {NUM_RUNS} validation runs...")
        print(f"Results will be saved to: {RESULTS_DIR}/")
        
        for i in range(1, NUM_RUNS + 1):
            run_test(i)
            if i < NUM_RUNS:
                print("\nWaiting 5 seconds before next run...")
                time.sleep(5)
        
        analyze_runs()
