#!/usr/bin/env python3
"""
Batch-Size Skalierungs-Test für KV-Cache Quantisierung.

Fragestellung: Wie skaliert der Speichervorteil bei parallelen Requests?

Misst:
- VRAM pro Batch-Size
- Throughput (Tokens/s) pro Batch
- Maximale Batch-Size bis OOM
"""

import sys
import time
import gc
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent))
from profiler import VRAMProfiler

from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class BatchResult:
    """Ergebnis eines Batch-Tests."""
    batch_size: int
    config: str
    vram_mb: float
    tokens_per_second: float
    total_tokens: int
    total_time_ms: float
    success: bool
    error: Optional[str] = None


def test_batch_size(
    model,
    tokenizer,
    profiler: VRAMProfiler,
    batch_size: int,
    config: dict,
    prompt: str,
    max_new_tokens: int = 50
) -> BatchResult:
    """Teste eine einzelne Batch-Size Konfiguration."""
    
    config_name = config["name"]
    
    # Prepare batched input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    # Expand to batch
    input_ids = input_ids.expand(batch_size, -1)
    attention_mask = torch.ones_like(input_ids)
    
    # Clear cache
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    try:
        start = time.perf_counter()
        
        with torch.no_grad():
            if config.get("quantize", False):
                output = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    cache_implementation="quantized",
                    cache_config={
                        "backend": config["backend"],
                        "nbits": config["nbits"]
                    }
                )
            else:
                output = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        # Measure peak VRAM
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        # Calculate throughput
        total_time_ms = (end - start) * 1000
        total_tokens = output.shape[0] * output.shape[1]  # batch × seq_len
        generated_tokens = batch_size * max_new_tokens
        tokens_per_second = generated_tokens / (total_time_ms / 1000)
        
        return BatchResult(
            batch_size=batch_size,
            config=config_name,
            vram_mb=peak_memory_mb,
            tokens_per_second=tokens_per_second,
            total_tokens=total_tokens,
            total_time_ms=total_time_ms,
            success=True
        )
        
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return BatchResult(
            batch_size=batch_size,
            config=config_name,
            vram_mb=0,
            tokens_per_second=0,
            total_tokens=0,
            total_time_ms=0,
            success=False,
            error="OOM"
        )
    except Exception as e:
        torch.cuda.empty_cache()
        return BatchResult(
            batch_size=batch_size,
            config=config_name,
            vram_mb=0,
            tokens_per_second=0,
            total_tokens=0,
            total_time_ms=0,
            success=False,
            error=str(e)
        )


def main():
    print("=" * 80)
    print("Batch-Size Skalierungs-Test - Mistral-7B")
    print("=" * 80)
    
    # Initialize profiler
    profiler = VRAMProfiler()
    
    # Load model
    model_name = "mistralai/Mistral-7B-v0.1"
    print(f"\nLoading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        low_cpu_mem_usage=True
    )
    model.eval()
    
    profiler.log_vram("Model Loaded")
    model_vram = profiler.measure_vram_mb()
    
    # Test configuration
    prompt = "The future of artificial intelligence in healthcare is"
    max_new_tokens = 50
    
    # Batch sizes to test (start small, increase until OOM)
    batch_sizes = [1, 2, 4, 8, 16, 32]
    
    configs = [
        {"name": "FP16", "quantize": False},
        {"name": "INT8", "quantize": True, "nbits": 8, "backend": "hqq"},
        {"name": "INT4", "quantize": True, "nbits": 4, "backend": "hqq"},
    ]
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Model VRAM: {model_vram:.0f} MB")
    print()
    
    # Results storage
    results: List[BatchResult] = []
    
    # Header
    print("=" * 100)
    print(f"{'Config':<10} {'Batch':>6} {'VRAM (MB)':>12} {'Tok/s':>10} {'Time (ms)':>12} {'Status':>10}")
    print("-" * 100)
    
    for config in configs:
        max_successful_batch = 0
        
        for batch_size in batch_sizes:
            result = test_batch_size(
                model=model,
                tokenizer=tokenizer,
                profiler=profiler,
                batch_size=batch_size,
                config=config,
                prompt=prompt,
                max_new_tokens=max_new_tokens
            )
            results.append(result)
            
            if result.success:
                max_successful_batch = batch_size
                status = "✅"
                print(f"{config['name']:<10} {batch_size:>6} {result.vram_mb:>12.0f} {result.tokens_per_second:>10.1f} {result.total_time_ms:>12.0f} {status:>10}")
            else:
                status = f"❌ {result.error}"
                print(f"{config['name']:<10} {batch_size:>6} {'-':>12} {'-':>10} {'-':>12} {status:>10}")
                break  # Stop testing larger batches for this config
        
        print()  # Separator between configs
    
    print("-" * 100)
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSE: VRAM-Ersparnis durch Quantisierung")
    print("=" * 80)
    
    # Compare VRAM at each batch size
    print(f"\n{'Batch':<8} {'FP16 (MB)':>12} {'INT8 (MB)':>12} {'INT4 (MB)':>12} {'INT8 Ersparnis':>15} {'INT4 Ersparnis':>15}")
    print("-" * 80)
    
    for batch_size in batch_sizes:
        fp16_result = next((r for r in results if r.batch_size == batch_size and r.config == "FP16" and r.success), None)
        int8_result = next((r for r in results if r.batch_size == batch_size and r.config == "INT8" and r.success), None)
        int4_result = next((r for r in results if r.batch_size == batch_size and r.config == "INT4" and r.success), None)
        
        fp16_vram = f"{fp16_result.vram_mb:.0f}" if fp16_result else "OOM"
        int8_vram = f"{int8_result.vram_mb:.0f}" if int8_result else "OOM"
        int4_vram = f"{int4_result.vram_mb:.0f}" if int4_result else "OOM"
        
        int8_save = ""
        int4_save = ""
        
        if fp16_result and int8_result:
            save_mb = fp16_result.vram_mb - int8_result.vram_mb
            int8_save = f"{save_mb:.0f} MB ({save_mb/fp16_result.vram_mb*100:.1f}%)"
        
        if fp16_result and int4_result:
            save_mb = fp16_result.vram_mb - int4_result.vram_mb
            int4_save = f"{save_mb:.0f} MB ({save_mb/fp16_result.vram_mb*100:.1f}%)"
        
        print(f"{batch_size:<8} {fp16_vram:>12} {int8_vram:>12} {int4_vram:>12} {int8_save:>15} {int4_save:>15}")
    
    print("-" * 80)
    
    # Find max batch sizes
    print("\n📊 MAXIMALE BATCH-SIZE:")
    for config in configs:
        max_batch = max((r.batch_size for r in results if r.config == config["name"] and r.success), default=0)
        print(f"   {config['name']}: {max_batch}")
    
    # Throughput scaling
    print("\n📊 THROUGHPUT-SKALIERUNG (Tokens/s):")
    print(f"{'Batch':<8} {'FP16':>12} {'INT8':>12} {'INT4':>12}")
    print("-" * 50)
    
    for batch_size in batch_sizes:
        fp16_result = next((r for r in results if r.batch_size == batch_size and r.config == "FP16" and r.success), None)
        int8_result = next((r for r in results if r.batch_size == batch_size and r.config == "INT8" and r.success), None)
        int4_result = next((r for r in results if r.batch_size == batch_size and r.config == "INT4" and r.success), None)
        
        fp16_tps = f"{fp16_result.tokens_per_second:.1f}" if fp16_result else "-"
        int8_tps = f"{int8_result.tokens_per_second:.1f}" if int8_result else "-"
        int4_tps = f"{int4_result.tokens_per_second:.1f}" if int4_result else "-"
        
        print(f"{batch_size:<8} {fp16_tps:>12} {int8_tps:>12} {int4_tps:>12}")
    
    print("\n✅ Batch-Size Test completed!")


if __name__ == "__main__":
    main()
