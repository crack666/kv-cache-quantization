#!/usr/bin/env python3
"""
Throughput-Test für Mistral-7B mit verschiedenen KV-Cache-Quantisierungen.

Testet:
- FP16 Baseline
- INT8 (HQQ)
- INT4 (HQQ)
- INT2 (HQQ)

Misst: Tokens/s, TTFT, TPT, VRAM
"""

import sys
import time
import torch
from pathlib import Path

# Add parent for profiler import
sys.path.insert(0, str(Path(__file__).parent))
from profiler import VRAMProfiler, ThroughputResult

from transformers import AutoModelForCausalLM, AutoTokenizer, QuantizedCache, DynamicCache


def test_throughput(
    model,
    tokenizer,
    profiler: VRAMProfiler,
    prompt: str,
    max_new_tokens: int,
    cache_config: dict,
    label: str
) -> ThroughputResult:
    """Run throughput test with specified cache configuration."""
    
    # Prepare input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    attention_mask = torch.ones_like(input_ids)
    prompt_tokens = input_ids.shape[-1]
    
    # Clear cache
    torch.cuda.empty_cache()
    profiler.log_vram(f"{label} - Before Generation")
    
    # Define generation function with cache_implementation parameter
    def generate():
        with torch.no_grad():
            if cache_config.get('quantize', False):
                # Use cache_implementation parameter (HF 4.43+)
                return model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    cache_implementation="quantized",
                    cache_config={
                        "backend": cache_config['backend'],
                        "nbits": cache_config['nbits']
                    }
                )
            else:
                return model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
    
    # Measure throughput
    result = profiler.measure_generation_throughput(
        generate_fn=generate,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        label=label,
        warmup_runs=1
    )
    
    profiler.log_vram(f"{label} - After Generation")
    
    return result


def main():
    print("=" * 70)
    print("Mistral-7B Throughput Test")
    print("=" * 70)
    
    # Initialize profiler
    profiler = VRAMProfiler()
    
    # Load model
    model_name = "mistralai/Mistral-7B-v0.1"
    print(f"\nLoading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        low_cpu_mem_usage=True
    )
    model.eval()
    
    profiler.log_vram("Model Loaded")
    
    # Test configuration
    prompt = "The future of artificial intelligence in healthcare is"
    max_new_tokens = 100
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Generating: {max_new_tokens} tokens per config\n")
    print("-" * 70)
    
    # Test configurations
    configs = [
        {"name": "FP16 Baseline", "quantize": False},
        {"name": "INT8 (HQQ)", "quantize": True, "nbits": 8, "backend": "hqq"},
        {"name": "INT4 (HQQ)", "quantize": True, "nbits": 4, "backend": "hqq"},
        {"name": "INT2 (HQQ)", "quantize": True, "nbits": 2, "backend": "hqq"},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{'='*70}")
        print(f"Testing: {config['name']}")
        print(f"{'='*70}")
        
        run_start = time.time()
        
        try:
            result = test_throughput(
                model=model,
                tokenizer=tokenizer,
                profiler=profiler,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                cache_config=config,
                label=config['name']
            )
            run_duration = time.time() - run_start
            print(f"⏱️  Run duration: {run_duration:.1f}s")
            results.append({"config": config['name'], "result": result, "duration_s": run_duration})
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({"config": config['name'], "error": str(e)})
    
    # Summary table with all metrics
    print("\n" + "=" * 100)
    print("THROUGHPUT SUMMARY - Mistral-7B (100 tokens)")
    print("=" * 100)
    print(f"{'Config':<18} {'Tok/s':>10} {'TTFT':>10} {'TPT':>10} {'Peak MB':>10} {'Watts':>8} {'mJ/tok':>10}")
    print("-" * 100)
    
    baseline_tps = None
    for r in results:
        if "error" in r:
            print(f"{r['config']:<18} {'ERROR':>10} {'-':>10} {'-':>10} {'-':>10} {'-':>8} {'-':>10}")
        else:
            result = r['result']
            tps = result.tokens_per_second
            
            if baseline_tps is None:
                baseline_tps = tps
                speedup = ""
            else:
                speedup = f"({tps/baseline_tps:.0%})"
            
            print(f"{r['config']:<18} {tps:>7.1f} {speedup:>3} {result.ttft_ms:>10.1f} {result.tpt_ms:>10.1f} {result.peak_memory_mb:>10.0f} {result.avg_power_watts:>8.0f} {result.energy_per_token_mj:>10.1f}")
    
    print("-" * 100)
    
    # Calculate efficiency metrics
    print("\nEFFICIENCY ANALYSIS:")
    if len(results) >= 2 and all("result" in r for r in results):
        baseline = results[0]["result"]
        for r in results[1:]:
            res = r["result"]
            throughput_ratio = res.tokens_per_second / baseline.tokens_per_second
            memory_ratio = res.peak_memory_mb / baseline.peak_memory_mb if baseline.peak_memory_mb > 0 else 1.0
            energy_ratio = res.energy_per_token_mj / baseline.energy_per_token_mj if baseline.energy_per_token_mj > 0 else 1.0
            print(f"  {r['config']}: {throughput_ratio:.0%} throughput, {memory_ratio:.0%} memory, {energy_ratio:.0%} energy vs FP16")
    
    # Save results
    profiler.save_to_json("throughput_mistral_test.json")
    
    print("\n✅ Throughput test completed!")


if __name__ == "__main__":
    main()
