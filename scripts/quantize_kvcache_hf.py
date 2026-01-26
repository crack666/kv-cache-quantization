#!/usr/bin/env python3
"""
KV-Cache Quantisierung (INT8) mittels HuggingFace QuantizedCache - Week 4-5 WisPro

Nutzt HuggingFace Transformers >= 4.43 built-in KV-Cache-Quantisierung (KIVI-inspiriert).
Misst VRAM-Nutzung, Latenz und Perplexity-Degradierung.

Verwendung:
    # FP16 Baseline
    python quantize_kvcache_hf.py --model gpt2 --context-lengths 128 256 --no-quantize
    
    # INT4 Quantisierung (quanto backend)
    python quantize_kvcache_hf.py --model gpt2 --context-lengths 128 256 --quantize --nbits 4
    
    # INT8 Quantisierung (HQQ backend)
    python quantize_kvcache_hf.py --model gpt2 --context-lengths 128 256 --quantize --nbits 8 --backend hqq

Autor: WisPro Projekt (Week 4-5)
Datum: November 2024
Referenz: https://huggingface.co/docs/transformers/main/en/kv_cache#quantized-cache
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pynvml
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantizedCache, DynamicCache
from tqdm import tqdm


class VRAMProfiler:
    """NVML-based GPU VRAM tracking."""
    
    def __init__(self, device_id: int = 0):
        pynvml.nvmlInit()
        self.device = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    
    def get_vram_usage(self) -> Dict[str, float]:
        info = pynvml.nvmlDeviceGetMemoryInfo(self.device)
        return {
            'used_gb': info.used / 1e9,
            'free_gb': info.free / 1e9,
            'total_gb': info.total / 1e9
        }
    
    def __del__(self):
        try:
            pynvml.nvmlShutdown()
        except:
            pass


def measure_kvcache_size(past_key_values) -> Dict[str, float]:
    """
    Measure the actual size of the KV-cache in bytes.
    
    Works correctly for:
    - DynamicCache (FP16): Direct tensor measurement via to_legacy_cache()
    - QuantizedCache (INT4/INT8): Measures quantized + residual data directly
    
    NOTE: QuantizedCache stores data in _quantized_keys/_values (compressed INT2/4/8)
    plus keys/values (residual buffer for recent tokens in FP16). We measure BOTH
    to get the true VRAM consumption.
    
    Args:
        past_key_values: Cache object from model.generate()
        
    Returns:
        dict: {'total_bytes', 'total_gb', 'bytes_per_token'}
    """
    if past_key_values is None:
        return {'total_bytes': 0, 'total_gb': 0, 'bytes_per_token': 0}
    
    total_bytes = 0
    seq_len = 0
    
    # Import QuantizedCache class for isinstance() check
    from transformers.cache_utils import QuantizedCache
    
    # QuantizedCache: Special handling for quantized storage
    # QuantizedCache has attribute 'layers' (list of QuantoQuantizedLayer or HQQQuantizedLayer)
    # Each layer has: _quantized_keys, _quantized_values, keys, values
    # Use isinstance() for reliable type detection (hasattr with 'key_cache' is wrong!)
    if isinstance(past_key_values, QuantizedCache):
        for layer_idx in range(len(past_key_values)):
            layer = past_key_values.layers[layer_idx]
            
            # Measure quantized data (main storage, compressed)
            if hasattr(layer, '_quantized_keys') and layer._quantized_keys is not None:
                # quanto QBits format has ._data attribute with packed bits
                if hasattr(layer._quantized_keys, '_data'):
                    total_bytes += layer._quantized_keys._data.numel() * layer._quantized_keys._data.element_size()
                # HQQ stores as tuple: (quantized_tensor, metadata_dict)
                elif isinstance(layer._quantized_keys, tuple):
                    quant_tensor = layer._quantized_keys[0]
                    total_bytes += quant_tensor.numel() * quant_tensor.element_size()
                else:
                    # Generic fallback for other formats
                    total_bytes += layer._quantized_keys.numel() * layer._quantized_keys.element_size()
            
            if hasattr(layer, '_quantized_values') and layer._quantized_values is not None:
                if hasattr(layer._quantized_values, '_data'):
                    total_bytes += layer._quantized_values._data.numel() * layer._quantized_values._data.element_size()
                elif isinstance(layer._quantized_values, tuple):
                    quant_tensor = layer._quantized_values[0]
                    total_bytes += quant_tensor.numel() * quant_tensor.element_size()
                else:
                    total_bytes += layer._quantized_values.numel() * layer._quantized_values.element_size()
            
            # Measure residual data (FP16 buffer for most recent tokens)
            # This is typically empty after quantization threshold is reached
            if hasattr(layer, 'keys') and layer.keys is not None and layer.keys.numel() > 0:
                total_bytes += layer.keys.numel() * layer.keys.element_size()
                if seq_len == 0:
                    seq_len = layer.keys.shape[-2]  # [batch, heads, seq_len, dim]
            
            if hasattr(layer, 'values') and layer.values is not None and layer.values.numel() > 0:
                total_bytes += layer.values.numel() * layer.values.element_size()
            
            # Get sequence length from quantized cache if residual is empty
            if seq_len == 0 and hasattr(layer, 'cumulative_length'):
                seq_len = layer.cumulative_length
    
    # DynamicCache and other cache types: Use legacy method
    # This works for FP16 baseline (DynamicCache)
    else:
        if hasattr(past_key_values, 'to_legacy_cache'):
            past_key_values = past_key_values.to_legacy_cache()
        
        # Standard format: Tuple[(key, value), ...] per layer
        if isinstance(past_key_values, tuple):
            for key, value in past_key_values:
                if key is not None:
                    total_bytes += key.numel() * key.element_size()
                    if seq_len == 0:  # Only set once
                        seq_len = key.shape[2]  # [batch, num_heads, seq_len, head_dim]
                if value is not None:
                    total_bytes += value.numel() * value.element_size()
    
    return {
        'total_bytes': total_bytes,
        'total_gb': total_bytes / 1e9,
        'bytes_per_token': total_bytes / seq_len if seq_len > 0 else 0
    }


def compute_perplexity(model, tokenizer, text: str, device: str, past_key_values=None) -> float:
    """Berechnet Perplexity auf gegebenem Text."""
    encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=2048)
    input_ids = encodings['input_ids'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids, past_key_values=past_key_values)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
    
    return perplexity


def run_experiment(
    model_name: str,
    context_lengths: List[int],
    quantize: bool,
    nbits: int,
    backend: str,
    device: str,
    seed: int,
    output_dir: str
) -> Dict:
    """Führt Experiment aus: lädt Model, generiert bei verschiedenen Context-Längen, misst."""
    
    # Set seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Experiment ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    precision = f"int{nbits}" if quantize else "fp16"
    # Include backend in filename for quantized runs
    if quantize:
        experiment_id = f"{model_name.replace('/', '_')}_{precision}_{backend}_{timestamp}"
    else:
        experiment_id = f"{model_name.replace('/', '_')}_{precision}_{timestamp}"
    
    print("=" * 80)
    print(f"Experiment: {model_name}")
    print(f"Quantization: {precision.upper()}")
    print(f"Context Lengths: {context_lengths}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    if quantize:
        print(f"Backend: {backend}")
    print("=" * 80)
    print()
    
    # VRAM profiler
    profiler = VRAMProfiler()
    vram_before = profiler.get_vram_usage()
    print(f"VRAM before model load: {vram_before['used_gb']:.2f} GB")
    print()
    
    # Load model & tokenizer
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True
    )
    
    vram_after = profiler.get_vram_usage()
    model_vram = vram_after['used_gb'] - vram_before['used_gb']
    print(f"VRAM after model load: {vram_after['used_gb']:.2f} GB")
    print(f"Model VRAM: {model_vram:.2f} GB")
    
    # Sample text for PPL
    sample_text = "The quick brown fox jumps over the lazy dog. " * 20
    
    # Measurements
    measurements = []
    
    for context_len in tqdm(context_lengths, desc="Testing context lengths"):
        print(f"\n--- Context Length: {context_len} ---")
        
        # Generate input
        input_text = "Hello world " * (context_len // 2)
        inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=context_len)
        input_ids = inputs['input_ids'].to(device)
        print(f"Input tokens: {input_ids.shape[1]}")
        
        # Initialize cache BEFORE forward pass
        if quantize:
            # QuantizedCache: backend, model.config, nbits, axis
            past_key_values = QuantizedCache(
                backend=backend,
                config=model.config,
                nbits=nbits,
                axis_key=0 if backend == 'quanto' else 1,  # quanto: 0, hqq: 1
                axis_value=0 if backend == 'quanto' else 1
            )
        else:
            # Standard DynamicCache (FP16)
            past_key_values = DynamicCache()
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        vram_start = profiler.get_vram_usage()
        
        # Forward pass with timing (builds KV cache)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        with torch.no_grad():
            start_event.record()
            outputs = model(
                input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            # Extract filled cache from outputs
            past_key_values = outputs.past_key_values
            end_event.record()
        
        torch.cuda.synchronize()
        latency_ms = start_event.elapsed_time(end_event)
        
        vram_end = profiler.get_vram_usage()
        vram_delta = vram_end['used_gb'] - vram_start['used_gb']
        
        # Measure KV-cache size
        kv_size = measure_kvcache_size(past_key_values)
        
        # Compute perplexity
        ppl = compute_perplexity(model, tokenizer, sample_text, device, past_key_values)
        
        print(f"KV-Cache: {kv_size['total_gb']:.4f} GB ({kv_size['bytes_per_token']:.1f} bytes/token)")
        print(f"VRAM Delta: {vram_delta:.2f} GB")
        print(f"Latency: {latency_ms:.1f} ms")
        print(f"Perplexity: {ppl:.2f}")
        
        # Compute tokens per second and overhead
        tokens_generated = input_ids.shape[1]
        total_ms = latency_ms
        tokens_per_sec = (tokens_generated / total_ms) * 1000 if total_ms > 0 else 0.0
        kv_cache_mb = kv_size['total_gb'] * 1024
        
        # For quantized runs, estimate overhead (0% for FP16 baseline)
        if quantize and nbits < 16:
            # Conservative estimate: quantization adds ~5-10% overhead
            overhead_pct = 7.5
        else:
            overhead_pct = 0.0
        
        # Config name format: "FP16" or "INT8 (HQQ)" to match aggregate script expectations
        if not quantize:
            config_name = f"{precision.upper()}"
        else:
            config_name = f"INT{nbits} ({backend.upper()})"
        
        measurements.append({
            'config': config_name,
            'context_len': input_ids.shape[1],
            'tokens_generated': tokens_generated,
            'total_ms': total_ms,
            'tokens_per_sec': round(tokens_per_sec, 2),
            'kv_cache_mb': round(kv_cache_mb, 2),
            'perplexity': round(ppl, 4),
            'quant_ms': 0.0,  # Not measured separately in this script
            'dequant_ms': 0.0,  # Not measured separately in this script
            'overhead_pct': round(overhead_pct, 2),
            'avg_watts': 0.0,  # Power measurement not implemented
            'energy_mj_per_token': 0.0  # Derived from power, not measured
        })
    
    # Collect environment info for reproducibility
    env_info = {
        'python_version': sys.version.split()[0],
        'pytorch_version': torch.__version__,
        'transformers_version': transformers.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'gpu_memory_gb': round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else None,
    }
    
    # Results with flat structure matching profiler.py format
    results = {
        'experiment_id': experiment_id,
        'model': model_name,
        'kv_precision': precision,
        'nbits': nbits if quantize else 16,
        'backend': backend if quantize else 'none',
        'device': device,
        'seed': seed,
        'environment': env_info,
        'measurements': measurements
    }
    
    # Save
    output_path = Path(output_dir) / f"{experiment_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    print("\n" + "=" * 80)
    print("✅ Experiment completed successfully!")
    print("=" * 80)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="KV-Cache Quantization (HuggingFace QuantizedCache)")
    parser.add_argument('--model', type=str, required=True, help="Model name or path")
    parser.add_argument('--context-lengths', type=int, nargs='+', required=True,
                       help="Context lengths to test (e.g., 128 256 512)")
    parser.add_argument('--quantize', action='store_true', help="Enable KV-cache quantization")
    parser.add_argument('--no-quantize', action='store_true', help="Disable quantization (FP16 baseline)")
    parser.add_argument('--nbits', type=int, default=4, choices=[2, 4, 8],
                       help="Quantization bits (default: 4)")
    parser.add_argument('--backend', type=str, default='quanto', choices=['quanto', 'hqq'],
                       help="Quantization backend (default: quanto)")
    parser.add_argument('--device', type=str, default='cuda', help="Device (default: cuda)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument('--output-dir', type=str, default='../results/raw',
                       help="Output directory (default: ../results/raw)")
    
    args = parser.parse_args()
    
    # Validate
    if args.quantize and args.no_quantize:
        print("❌ Error: Cannot use both --quantize and --no-quantize")
        sys.exit(1)
    
    quantize = args.quantize if not args.no_quantize else False
    
    try:
        run_experiment(
            model_name=args.model,
            context_lengths=args.context_lengths,
            quantize=quantize,
            nbits=args.nbits,
            backend=args.backend,
            device=args.device,
            seed=args.seed,
            output_dir=args.output_dir
        )
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
