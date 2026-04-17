#!/usr/bin/env python3
"""
Detailliertes Profiling der Quantisierungs-Overhead.

Instrumentiert die HuggingFace QuantizedCache-Klassen um exakte Zeitmessungen
für Quantisierung und Dequantisierung zu erhalten.

Ergebnis: Wie viel Zeit verbringen wir mit Quant/Dequant vs. Attention?

Usage:
    python profile_quant_overhead.py                                    # Default: Mistral-7B, contexts 128-4096
    python profile_quant_overhead.py --model Qwen/Qwen3-8B              # Custom model
    python profile_quant_overhead.py --context 512 1024 2048            # Specific contexts only
    python profile_quant_overhead.py --output results/my_profile.json   # Custom output path
"""

import argparse
import time
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from contextlib import contextmanager


def measure_kv_cache_size(past_key_values) -> Tuple[float, str]:
    """
    Berechne exakte KV-Cache-Größe aus PyTorch-Tensoren.
    
    WICHTIG: Bei QuantizedCache (HQQ/Quanto) sind die echten quantisierten Daten
    in `_quantized_keys` und `_quantized_values` gespeichert (als Tuple: qtensor, meta).
    Die `.keys/.values` Attribute enthalten nur dequantisierte FP16-Daten für Attention!
    
    Unterstützt:
    - DynamicCache (FP16, iterierbar)
    - QuantizedCache mit HQQ backend (_quantized_keys/_quantized_values)
    - Legacy tuple format
    
    Returns: (size_mb, cache_type)
    """
    if past_key_values is None:
        return 0.0, "None"
    
    kv_cache_bytes = 0
    cache_type = type(past_key_values).__name__
    
    # Check for QuantizedCache with .layers attribute (HQQ/Quanto)
    if hasattr(past_key_values, 'layers') and len(past_key_values.layers) > 0:
        layer0 = past_key_values.layers[0]
        
        # HQQQuantizedLayer: echte quantisierte Daten in _quantized_keys/_quantized_values
        if hasattr(layer0, '_quantized_keys'):
            for layer in past_key_values.layers:
                # Quantized keys: (qtensor, meta) tuple
                if hasattr(layer, '_quantized_keys') and layer._quantized_keys is not None:
                    qtensor, meta = layer._quantized_keys
                    kv_cache_bytes += qtensor.element_size() * qtensor.numel()
                    # Scale und Zero-Point (FP16) auch zählen
                    kv_cache_bytes += meta['scale'].element_size() * meta['scale'].numel()
                    kv_cache_bytes += meta['zero'].element_size() * meta['zero'].numel()
                
                # Quantized values: (qtensor, meta) tuple
                if hasattr(layer, '_quantized_values') and layer._quantized_values is not None:
                    qtensor, meta = layer._quantized_values
                    kv_cache_bytes += qtensor.element_size() * qtensor.numel()
                    kv_cache_bytes += meta['scale'].element_size() * meta['scale'].numel()
                    kv_cache_bytes += meta['zero'].element_size() * meta['zero'].numel()
                
                # Residual (unquantisierte letzte Tokens) in .keys/.values
                if hasattr(layer, 'keys') and layer.keys is not None and layer.keys.numel() > 0:
                    kv_cache_bytes += layer.keys.element_size() * layer.keys.numel()
                if hasattr(layer, 'values') and layer.values is not None and layer.values.numel() > 0:
                    kv_cache_bytes += layer.values.element_size() * layer.values.numel()
            
            nbits = getattr(layer0, 'nbits', '?')
            return kv_cache_bytes / (1024 * 1024), f"QuantizedCache (INT{nbits})"
        
        # DynamicCache mit .layers (neuere HF Version) - FP16
        elif hasattr(layer0, 'keys') and hasattr(layer0, 'values'):
            for layer in past_key_values.layers:
                if layer.keys is not None:
                    kv_cache_bytes += layer.keys.element_size() * layer.keys.numel()
                if layer.values is not None:
                    kv_cache_bytes += layer.values.element_size() * layer.values.numel()
            return kv_cache_bytes / (1024 * 1024), "DynamicCache (FP16)"
    
    # Legacy: Try iterating (works for older DynamicCache and tuple format)
    try:
        for layer_idx, layer_kv in enumerate(past_key_values):
            if isinstance(layer_kv, tuple) and len(layer_kv) >= 2:
                key_tensor, value_tensor = layer_kv[0], layer_kv[1]
                
                if isinstance(key_tensor, torch.Tensor):
                    kv_cache_bytes += key_tensor.element_size() * key_tensor.numel()
                if isinstance(value_tensor, torch.Tensor):
                    kv_cache_bytes += value_tensor.element_size() * value_tensor.numel()
        
        if kv_cache_bytes > 0:
            return kv_cache_bytes / (1024 * 1024), "DynamicCache (FP16, legacy)"
    except (TypeError, IndexError):
        pass
    
    return 0.0, cache_type + " (unknown structure)"


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
    import gc
    import threading
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Power monitoring via NVML
    try:
        import pynvml
        pynvml.nvmlInit()
        GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
        NVML_AVAILABLE = True
    except:
        NVML_AVAILABLE = False
        print("Warning: pynvml not available, power monitoring disabled")
    
    @dataclass
    class FullMetrics:
        """Alle Metriken für einen Run."""
        config: str
        context_len: int
        tokens_generated: int
        total_ms: float
        tokens_per_sec: float
        kv_cache_mb: float
        perplexity: float  # NEW: Qualitätsmetrik
        quant_ms: float
        dequant_ms: float
        overhead_pct: float
        avg_watts: float
        energy_mj_per_token: float
    
    def sample_power_background(stop_event, power_samples):
        """Background thread für Power-Sampling."""
        while not stop_event.is_set():
            if NVML_AVAILABLE:
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(GPU_HANDLE)
                    power_samples.append(power_mw / 1000.0)  # mW -> W
                except:
                    pass
            time.sleep(0.01)  # 100 Hz sampling
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="KV-Cache Quantization Profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python profile_quant_overhead.py
  python profile_quant_overhead.py --model Qwen/Qwen2-7B
  python profile_quant_overhead.py --context 512 1024 2048
  python profile_quant_overhead.py --model Qwen/Qwen3-8B --context 128 256 --output my_results.json
        """)
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1",
                        help="HuggingFace model name (default: Mistral-7B-v0.1)")
    parser.add_argument("--context", type=int, nargs="+", default=None,
                        help="Target context lengths (default: 128 256 512 1024 2048 4096)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: results/raw/profile_<model>_<timestamp>.json)")
    args = parser.parse_args()
    
    model_name = args.model
    
    print("=" * 120)
    print(f"VOLLSTÄNDIGES PROFILING - {model_name} - Alle Metriken")
    print("=" * 120)
    
    # Apply patches BEFORE loading model
    patch_quantized_cache()
    
    # Load model (now from CLI argument)
    print(f"\nLoading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        low_cpu_mem_usage=True
    )
    model.eval()
    
    # Measure baseline VRAM
    torch.cuda.synchronize()
    model_vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"Model VRAM: {model_vram_mb:.0f} MB")
    
    # Target context lengths (2er-Potenzen für wissenschaftliche Konsistenz)
    # max_new_tokens = target_context - prompt_tokens
    target_contexts = args.context if args.context else [128, 256, 512, 1024, 2048, 4096]
    configs = [
        {"name": "FP16", "quantize": False},
        {"name": "INT8 (HQQ)", "quantize": True, "nbits": 8, "backend": "hqq"},
        {"name": "INT4 (HQQ)", "quantize": True, "nbits": 4, "backend": "hqq"},
        {"name": "INT2 (HQQ)", "quantize": True, "nbits": 2, "backend": "hqq"},
    ]
    
    prompt_base = "The future of artificial intelligence in healthcare is "
    
    all_results: List[FullMetrics] = []
    
    # Compute perplexity on generated text
    def compute_perplexity(model, tokenizer, text: str) -> float:
        """Berechnet Perplexity auf generiertem Text."""
        if len(text.strip()) < 10:
            return float('inf')
        encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=2048)
        input_ids = encodings['input_ids'].to('cuda')
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        return perplexity
    
    # Get prompt token count
    prompt_tokens = tokenizer(prompt_base, return_tensors="pt").input_ids.shape[1]
    print(f"Prompt tokens: {prompt_tokens}")
    
    # Header - mit PPL-Spalte
    print("\n" + "=" * 190)
    print(f"{'Config':<12} {'TgtCtx':>6} {'GenTok':>8} {'Time ms':>10} {'Tok/s':>8} {'KV MB':>10} {'PPL':>8} {'Quant ms':>10} {'Dequant ms':>11} {'Overhead':>9} {'Watts':>8} {'mJ/tok':>8}")
    print("-" * 190)
    
    for target_ctx in target_contexts:
        # Calculate max_new_tokens to reach target context
        max_new_tokens = target_ctx - prompt_tokens
        for config in configs:
            # Prepare input
            input_ids = tokenizer(prompt_base, return_tensors="pt").input_ids.to("cuda")
            attention_mask = torch.ones_like(input_ids)
            
            # Reset everything
            reset_timings()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
            # Start power sampling
            power_samples = []
            stop_event = threading.Event()
            power_thread = threading.Thread(target=sample_power_background, args=(stop_event, power_samples))
            power_thread.start()
            
            try:
                # Measure total time
                start = time.perf_counter()
                
                with torch.no_grad():
                    if config["quantize"]:
                        output = model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=max_new_tokens,
                            min_new_tokens=max_new_tokens,  # Prevent early EOS stopping
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                            cache_implementation="quantized",
                            cache_config={
                                "backend": config["backend"],
                                "nbits": config["nbits"]
                            },
                            return_dict_in_generate=True,
                        )
                    else:
                        output = model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=max_new_tokens,
                            min_new_tokens=max_new_tokens,  # Prevent early EOS stopping
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                            return_dict_in_generate=True,
                        )
                
                torch.cuda.synchronize()
                total_ms = (time.perf_counter() - start) * 1000
                
                # Stop power sampling
                stop_event.set()
                power_thread.join()
                
                # Measure KV-Cache size from past_key_values
                kv_cache_mb = 0.0
                if hasattr(output, 'past_key_values') and output.past_key_values is not None:
                    kv_cache_mb, cache_type = measure_kv_cache_size(output.past_key_values)
                
                # Calculate metrics
                sequences = output.sequences if hasattr(output, 'sequences') else output
                tokens_generated = sequences.shape[1] - input_ids.shape[1]
                tokens_per_sec = tokens_generated / (total_ms / 1000)
                
                # Compute perplexity on generated text
                generated_text = tokenizer.decode(sequences[0][input_ids.shape[1]:], skip_special_tokens=True)
                perplexity = compute_perplexity(model, tokenizer, generated_text)
                
                # Timing breakdown
                timings = get_timings().summary()
                quant_ms = timings["quantize_total_ms"]
                dequant_ms = timings["dequantize_total_ms"]
                overhead_ms = quant_ms + dequant_ms
                overhead_pct = (overhead_ms / total_ms * 100) if total_ms > 0 else 0
                
                # Power metrics
                avg_watts = sum(power_samples) / len(power_samples) if power_samples else 0
                total_energy_j = avg_watts * (total_ms / 1000)  # Joules
                energy_mj_per_token = (total_energy_j * 1000) / tokens_generated if tokens_generated > 0 else 0
                
                result = FullMetrics(
                    config=config["name"],
                    context_len=target_ctx,
                    tokens_generated=tokens_generated,
                    total_ms=total_ms,
                    tokens_per_sec=tokens_per_sec,
                    kv_cache_mb=kv_cache_mb,
                    perplexity=perplexity,
                    quant_ms=quant_ms,
                    dequant_ms=dequant_ms,
                    overhead_pct=overhead_pct,
                    avg_watts=avg_watts,
                    energy_mj_per_token=energy_mj_per_token
                )
                all_results.append(result)
                
                print(f"{config['name']:<12} {target_ctx:>6} {tokens_generated:>8} {total_ms:>10.0f} {tokens_per_sec:>8.1f} {kv_cache_mb:>10.1f} {perplexity:>8.2f} {quant_ms:>10.1f} {dequant_ms:>11.1f} {overhead_pct:>8.1f}% {avg_watts:>8.0f} {energy_mj_per_token:>8.1f}")
                
            except torch.cuda.OutOfMemoryError:
                stop_event.set()
                power_thread.join()
                torch.cuda.empty_cache()
                print(f"{config['name']:<12} {target_ctx:>6} {'-':>8} {'OOM':>10} {'-':>8} {'-':>10} {'-':>8} {'-':>10} {'-':>11} {'-':>9} {'-':>8} {'-':>8}")
                break  # Skip remaining context lengths for this config
        
        print()  # Separator between context lengths
    
    print("-" * 160)
    
    # Summary analysis
    print("\n" + "=" * 80)
    print("ZUSAMMENFASSUNG")
    print("=" * 80)
    
    # Group by context length and compare KV-Cache sizes
    print("\n--- KV-CACHE GRÖSSE (direkt gemessen) ---")
    print(f"{'Ctx':>6} {'FP16 MB':>10} {'INT8 MB':>10} {'INT4 MB':>10} {'INT8 Ratio':>12} {'INT4 Ratio':>12}")
    print("-" * 70)
    
    for target_ctx in target_contexts:
        fp16 = next((r for r in all_results if r.context_len == target_ctx and r.config == "FP16"), None)
        int8 = next((r for r in all_results if r.context_len == target_ctx and r.config == "INT8 (HQQ)"), None)
        int4 = next((r for r in all_results if r.context_len == target_ctx and r.config == "INT4 (HQQ)"), None)
        
        if fp16 and fp16.kv_cache_mb > 0:
            fp16_mb = f"{fp16.kv_cache_mb:.1f}"
            int8_mb = f"{int8.kv_cache_mb:.1f}" if int8 else "-"
            int4_mb = f"{int4.kv_cache_mb:.1f}" if int4 else "-"
            
            int8_ratio = f"{int8.kv_cache_mb / fp16.kv_cache_mb * 100:.0f}%" if int8 and fp16.kv_cache_mb > 0 else "-"
            int4_ratio = f"{int4.kv_cache_mb / fp16.kv_cache_mb * 100:.0f}%" if int4 and fp16.kv_cache_mb > 0 else "-"
            
            print(f"{target_ctx:>6} {fp16_mb:>10} {int8_mb:>10} {int4_mb:>10} {int8_ratio:>12} {int4_ratio:>12}")
        else:
            print(f"{target_ctx:>6} {'(no data)':>10}")
    
    print("\n--- Throughput-Verhältnis (vs FP16) ---")
    print(f"{'Ctx':>6} {'FP16 Tok/s':>12} {'INT8':>10} {'INT4':>10} {'INT2':>10}")
    print("-" * 55)
    
    for target_ctx in target_contexts:
        fp16 = next((r for r in all_results if r.context_len == target_ctx and r.config == "FP16"), None)
        int8 = next((r for r in all_results if r.context_len == target_ctx and r.config == "INT8 (HQQ)"), None)
        int4 = next((r for r in all_results if r.context_len == target_ctx and r.config == "INT4 (HQQ)"), None)
        int2 = next((r for r in all_results if r.context_len == target_ctx and r.config == "INT2 (HQQ)"), None)
        
        if fp16:
            fp16_tps = f"{fp16.tokens_per_sec:.1f}"
            int8_ratio = f"{int8.tokens_per_sec / fp16.tokens_per_sec * 100:.0f}%" if int8 else "-"
            int4_ratio = f"{int4.tokens_per_sec / fp16.tokens_per_sec * 100:.0f}%" if int4 else "-"
            int2_ratio = f"{int2.tokens_per_sec / fp16.tokens_per_sec * 100:.0f}%" if int2 else "-"
            
            print(f"{target_ctx:>6} {fp16_tps:>12} {int8_ratio:>10} {int4_ratio:>10} {int2_ratio:>10}")
    
    print("\n--- Overhead bleibt konstant? ---")
    print(f"{'Ctx':>6} {'INT8 %':>10} {'INT4 %':>10} {'INT2 %':>10}")
    print("-" * 45)
    
    for target_ctx in target_contexts:
        int8 = next((r for r in all_results if r.context_len == target_ctx and r.config == "INT8 (HQQ)"), None)
        int4 = next((r for r in all_results if r.context_len == target_ctx and r.config == "INT4 (HQQ)"), None)
        int2 = next((r for r in all_results if r.context_len == target_ctx and r.config == "INT2 (HQQ)"), None)
        
        int8_oh = f"{int8.overhead_pct:.1f}%" if int8 else "-"
        int4_oh = f"{int4.overhead_pct:.1f}%" if int4 else "-"
        int2_oh = f"{int2.overhead_pct:.1f}%" if int2 else "-"
        
        print(f"{target_ctx:>6} {int8_oh:>10} {int4_oh:>10} {int2_oh:>10}")
    
    # NEW: PPL comparison
    print("\n--- Perplexity (Qualitätsverlust durch Quantisierung) ---")
    print(f"{'Ctx':>6} {'FP16 PPL':>10} {'INT8 PPL':>10} {'INT4 PPL':>10} {'INT2 PPL':>10} {'INT8 Δ%':>10} {'INT4 Δ%':>10}")
    print("-" * 80)
    
    for target_ctx in target_contexts:
        fp16 = next((r for r in all_results if r.context_len == target_ctx and r.config == "FP16"), None)
        int8 = next((r for r in all_results if r.context_len == target_ctx and r.config == "INT8 (HQQ)"), None)
        int4 = next((r for r in all_results if r.context_len == target_ctx and r.config == "INT4 (HQQ)"), None)
        int2 = next((r for r in all_results if r.context_len == target_ctx and r.config == "INT2 (HQQ)"), None)
        
        if fp16:
            fp16_ppl = f"{fp16.perplexity:.2f}"
            int8_ppl = f"{int8.perplexity:.2f}" if int8 else "-"
            int4_ppl = f"{int4.perplexity:.2f}" if int4 else "-"
            int2_ppl = f"{int2.perplexity:.2f}" if int2 else "-"
            
            int8_delta = f"+{(int8.perplexity / fp16.perplexity - 1) * 100:.1f}%" if int8 and fp16.perplexity > 0 else "-"
            int4_delta = f"+{(int4.perplexity / fp16.perplexity - 1) * 100:.1f}%" if int4 and fp16.perplexity > 0 else "-"
            
            print(f"{target_ctx:>6} {fp16_ppl:>10} {int8_ppl:>10} {int4_ppl:>10} {int2_ppl:>10} {int8_delta:>10} {int4_delta:>10}")
    
    # Save results to JSON for future analysis
    import json
    from datetime import datetime
    
    results_dict = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "prompt_tokens": prompt_tokens,
        "target_contexts": target_contexts,
        "configs": [c["name"] for c in configs],
        "measurements": [
            {
                "config": r.config,
                "context_len": r.context_len,
                "tokens_generated": r.tokens_generated,
                "total_ms": r.total_ms,
                "tokens_per_sec": r.tokens_per_sec,
                "kv_cache_mb": r.kv_cache_mb,
                "perplexity": r.perplexity,
                "quant_ms": r.quant_ms,
                "dequant_ms": r.dequant_ms,
                "overhead_pct": r.overhead_pct,
                "avg_watts": r.avg_watts,
                "energy_mj_per_token": r.energy_mj_per_token
            }
            for r in all_results
        ]
    }
    
    # Determine output path
    if args.output:
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        results_dir = Path(__file__).parent.parent / "results" / "raw"
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = model_name.split("/")[-1].lower().replace("-", "_")
        output_file = results_dir / f"profile_{model_short}_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print("Profiling complete!")
    print("=" * 80)
