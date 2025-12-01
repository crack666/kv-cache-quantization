#!/usr/bin/env python3
"""
Detailliertes Profiling der Quantisierungs-Overhead.

Instrumentiert die HuggingFace QuantizedCache-Klassen um exakte Zeitmessungen
für Quantisierung und Dequantisierung zu erhalten.

Ergebnis: Wie viel Zeit verbringen wir mit Quant/Dequant vs. Attention?
"""

import time
import torch
from dataclasses import dataclass, field
from typing import List, Dict
from contextlib import contextmanager


@dataclass
class QuantizationTimings:
    """Sammelt Timing-Daten für Quantisierungs-Operationen."""
    quantize_times_ms: List[float] = field(default_factory=list)
    dequantize_times_ms: List[float] = field(default_factory=list)
    
    def add_quantize(self, ms: float):
        self.quantize_times_ms.append(ms)
    
    def add_dequantize(self, ms: float):
        self.dequantize_times_ms.append(ms)
    
    def summary(self) -> Dict:
        q_total = sum(self.quantize_times_ms)
        dq_total = sum(self.dequantize_times_ms)
        return {
            "quantize_total_ms": round(q_total, 3),
            "quantize_calls": len(self.quantize_times_ms),
            "quantize_avg_ms": round(q_total / len(self.quantize_times_ms), 3) if self.quantize_times_ms else 0,
            "dequantize_total_ms": round(dq_total, 3),
            "dequantize_calls": len(self.dequantize_times_ms),
            "dequantize_avg_ms": round(dq_total / len(self.dequantize_times_ms), 3) if self.dequantize_times_ms else 0,
            "total_overhead_ms": round(q_total + dq_total, 3),
        }
    
    def reset(self):
        self.quantize_times_ms.clear()
        self.dequantize_times_ms.clear()


# Global timing collector
_timings = QuantizationTimings()


def get_timings() -> QuantizationTimings:
    return _timings


def reset_timings():
    _timings.reset()


def patch_quantized_cache():
    """Monkey-patch HuggingFace QuantizedCache um Timing zu messen."""
    from transformers.cache_utils import QuantoQuantizedLayer, HQQQuantizedLayer
    
    # Patch QuantoQuantizedLayer
    original_quanto_quantize = QuantoQuantizedLayer._quantize
    original_quanto_dequantize = QuantoQuantizedLayer._dequantize
    
    def timed_quanto_quantize(self, tensor, axis):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = original_quanto_quantize(self, tensor, axis)
        torch.cuda.synchronize()
        _timings.add_quantize((time.perf_counter() - start) * 1000)
        return result
    
    def timed_quanto_dequantize(self, qtensor):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = original_quanto_dequantize(self, qtensor)
        torch.cuda.synchronize()
        _timings.add_dequantize((time.perf_counter() - start) * 1000)
        return result
    
    QuantoQuantizedLayer._quantize = timed_quanto_quantize
    QuantoQuantizedLayer._dequantize = timed_quanto_dequantize
    
    # Patch HQQQuantizedLayer
    original_hqq_quantize = HQQQuantizedLayer._quantize
    original_hqq_dequantize = HQQQuantizedLayer._dequantize
    
    def timed_hqq_quantize(self, tensor, axis):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = original_hqq_quantize(self, tensor, axis)
        torch.cuda.synchronize()
        _timings.add_quantize((time.perf_counter() - start) * 1000)
        return result
    
    def timed_hqq_dequantize(self, qtensor):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = original_hqq_dequantize(self, qtensor)
        torch.cuda.synchronize()
        _timings.add_dequantize((time.perf_counter() - start) * 1000)
        return result
    
    HQQQuantizedLayer._quantize = timed_hqq_quantize
    HQQQuantizedLayer._dequantize = timed_hqq_dequantize
    
    print("✅ QuantizedCache timing patches applied")


@contextmanager
def measure_quantization_overhead():
    """Context manager für Overhead-Messung."""
    reset_timings()
    yield _timings
    # Timings sind jetzt in _timings verfügbar


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from profiler import VRAMProfiler
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("=" * 70)
    print("Quantisierungs-Overhead Profiling - Mistral-7B")
    print("=" * 70)
    
    # Apply patches BEFORE loading model
    patch_quantized_cache()
    
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
    
    # Test different context lengths
    context_lengths = [100, 500, 1000, 2000]
    configs = [
        {"name": "FP16 Baseline", "quantize": False},
        {"name": "INT8 (HQQ)", "quantize": True, "nbits": 8, "backend": "hqq"},
        {"name": "INT4 (HQQ)", "quantize": True, "nbits": 4, "backend": "hqq"},
    ]
    
    prompt_base = "The future of artificial intelligence in healthcare is "
    
    print("\n" + "=" * 100)
    print(f"{'Config':<18} {'Tokens':>8} {'Total ms':>12} {'Quant ms':>12} {'Dequant ms':>12} {'Overhead %':>12}")
    print("-" * 100)
    
    for max_new_tokens in context_lengths:
        for config in configs:
            # Prepare input
            input_ids = tokenizer(prompt_base, return_tensors="pt").input_ids.to("cuda")
            attention_mask = torch.ones_like(input_ids)
            
            # Reset timings
            reset_timings()
            
            # Warmup
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Measure total time
            start = time.perf_counter()
            
            with torch.no_grad():
                if config["quantize"]:
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
            total_ms = (time.perf_counter() - start) * 1000
            
            # Get timing breakdown
            timings = get_timings().summary()
            overhead_ms = timings["total_overhead_ms"]
            overhead_pct = (overhead_ms / total_ms * 100) if total_ms > 0 else 0
            
            print(f"{config['name']:<18} {max_new_tokens:>8} {total_ms:>12.1f} {timings['quantize_total_ms']:>12.1f} {timings['dequantize_total_ms']:>12.1f} {overhead_pct:>11.1f}%")
    
    print("-" * 100)
    print("\n✅ Profiling complete!")
    print("\nInterpretation:")
    print("  - Overhead % zeigt, wie viel Zeit für Quant/Dequant vs. Gesamtzeit aufgewendet wird")
    print("  - Bei kurzen Kontexten: Overhead ist signifikant → FP16 schneller")
    print("  - Bei langen Kontexten: Memory-Bandwidth wird Bottleneck → Quantisierung lohnt sich")
