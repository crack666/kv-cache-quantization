#!/usr/bin/env python3
"""
KV-Cache INT8 Quantization Script
==================================

Implementiert symmetric per-tensor INT8-Quantisierung für den KV-Cache von LLMs.

Author: Lennart Behr
Date: 2025-11-24
Hardware: RTX 5090 (32 GB VRAM)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pynvml
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# ============================================================================
# 1. VRAM Profiling Utilities
# ============================================================================

class VRAMProfiler:
    """Tracks GPU VRAM usage using NVML (NVIDIA Management Library)."""
    
    def __init__(self, device_id: int = 0):
        """
        Initialize NVML and select GPU device.
        
        Args:
            device_id: GPU device index (default: 0 for first GPU)
        """
        pynvml.nvmlInit()
        self.device = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        self.device_name = pynvml.nvmlDeviceGetName(self.device)
        
    def get_vram_usage(self) -> Dict[str, float]:
        """
        Query current VRAM usage.
        
        Returns:
            Dictionary with 'used_gb', 'free_gb', 'total_gb'
        """
        info = pynvml.nvmlDeviceGetMemoryInfo(self.device)
        return {
            'used_gb': info.used / 1e9,
            'free_gb': info.free / 1e9,
            'total_gb': info.total / 1e9
        }
    
    def __del__(self):
        """Cleanup NVML on object destruction."""
        try:
            pynvml.nvmlShutdown()
        except:
            pass


# ============================================================================
# 2. INT8 Quantization Core
# ============================================================================

class SymmetricINT8Quantizer:
    """
    Symmetric per-tensor INT8 quantization.
    
    Formula:
        scale = max(abs(tensor)) / 127
        quantized = round(tensor / scale).clamp(-128, 127).to(int8)
        dequantized = quantized.to(float16) * scale
    
    Rationale:
        - Symmetric: Zero-point is 0 (keine Asymmetrie)
        - Per-tensor: Ein Scale-Wert für gesamten Tensor (einfach, schnell)
        - INT8 Range: [-128, 127] (signed int8)
    """
    
    @staticmethod
    def quantize(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize FP16 tensor to INT8.
        
        Args:
            tensor: Input tensor (FP16 or FP32)
            
        Returns:
            (quantized_tensor, scale)
            - quantized_tensor: INT8 tensor
            - scale: FP32 scalar (for dequantization)
        """
        # Scale-Berechnung: max_abs_value / 127
        # Warum 127? INT8 range ist [-128, 127], aber wir nutzen symmetric [-127, 127]
        max_val = tensor.abs().max()
        
        # Edge case: tensor ist all zeros
        if max_val == 0:
            return torch.zeros_like(tensor, dtype=torch.int8), torch.tensor(1.0)
        
        scale = max_val / 127.0
        
        # Quantisierung: tensor / scale, dann round und clamp zu [-128, 127]
        quantized = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
        
        return quantized, scale
    
    @staticmethod
    def dequantize(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Dequantize INT8 tensor back to FP16.
        
        Args:
            quantized: INT8 tensor
            scale: FP32 scalar
            
        Returns:
            Dequantized FP16 tensor
        """
        return quantized.to(torch.float16) * scale


# ============================================================================
# 3. KV-Cache Quantization Hook
# ============================================================================

class KVCacheQuantizationHook:
    """
    HuggingFace-kompatible Hook-Klasse für KV-Cache-Quantisierung.
    
    Wird als `forward_hook` registriert und intercepted den KV-Cache
    nach jedem Attention-Layer.
    """
    
    def __init__(self, quantizer: SymmetricINT8Quantizer, enable: bool = True):
        """
        Args:
            quantizer: Quantisierungs-Instanz
            enable: Quantisierung aktivieren (zum Testen FP16 vs INT8)
        """
        self.quantizer = quantizer
        self.enable = enable
        self.scales_keys = []  # Speichert scales pro Layer (für Keys)
        self.scales_values = []  # Speichert scales pro Layer (für Values)
        
    def __call__(self, module, input, output):
        """
        Forward-Hook: Wird nach jedem Attention-Layer aufgerufen.
        
        Args:
            module: Attention-Layer (nn.Module)
            input: Input-Tuple
            output: Output-Tuple, enthält past_key_values
            
        Returns:
            Modified output mit quantisierten KV-Cache
        """
        if not self.enable:
            return output
        
        # HuggingFace Output: (logits, past_key_values)
        # past_key_values: Tuple[(key, value), ...] pro Layer
        if len(output) < 2 or output[1] is None:
            return output
        
        past_key_values = output[1]
        quantized_kv = []
        
        for layer_idx, (key, value) in enumerate(past_key_values):
            # Quantize Keys
            key_q, scale_k = self.quantizer.quantize(key)
            
            # Quantize Values
            value_q, scale_v = self.quantizer.quantize(value)
            
            # Speichere scales (für spätere Analyse)
            if len(self.scales_keys) <= layer_idx:
                self.scales_keys.append([])
                self.scales_values.append([])
            self.scales_keys[layer_idx].append(scale_k.item())
            self.scales_values[layer_idx].append(scale_v.item())
            
            # HINWEIS: HuggingFace erwartet FP16 tensors im KV-cache.
            # Für echte INT8-Speicherung bräuchten wir custom attention implementation.
            # WisPro-Ansatz: Dequantisiere für Model, aber tracke INT8-Größe separat.
            key_dq = self.quantizer.dequantize(key_q, scale_k)
            value_dq = self.quantizer.dequantize(value_q, scale_v)
            
            quantized_kv.append((key_dq, value_dq))
        
        # Return modified output
        return (output[0], tuple(quantized_kv))


# ============================================================================
# 4. Measurement Pipeline
# ============================================================================

def measure_kvcache_size(past_key_values) -> Dict[str, float]:
    """
    Berechnet KV-Cache-Größe aus past_key_values Tuple.
    
    Args:
        past_key_values: Tuple[(key, value), ...] pro Layer
        
    Returns:
        Dictionary mit 'total_bytes', 'total_gb', 'bytes_per_token'
    """
    if past_key_values is None:
        return {'total_bytes': 0, 'total_gb': 0, 'bytes_per_token': 0}
    
    total_bytes = 0
    for key, value in past_key_values:
        total_bytes += key.numel() * key.element_size()
        total_bytes += value.numel() * value.element_size()
    
    # Bytes per token: total_bytes / sequence_length
    # Wir nehmen sequence_length von key.shape[2] (batch, num_heads, seq_len, head_dim)
    seq_len = past_key_values[0][0].shape[2] if len(past_key_values) > 0 else 1
    
    return {
        'total_bytes': total_bytes,
        'total_gb': total_bytes / 1e9,
        'bytes_per_token': total_bytes / seq_len if seq_len > 0 else 0
    }


def compute_perplexity(model, tokenizer, text: str, device: str) -> float:
    """
    Berechnet Perplexity auf gegebenem Text.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        text: Input text
        device: 'cuda' or 'cpu'
        
    Returns:
        Perplexity (exp(cross_entropy_loss))
    """
    encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=2048)
    input_ids = encodings['input_ids'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
    
    return perplexity


def run_experiment(
    model_name: str,
    context_lengths: List[int],
    quantize: bool,
    output_dir: Path,
    device: str = 'cuda',
    seed: int = 42
) -> Dict:
    """
    Führt Experiment durch: FP16 vs INT8 KV-Cache Messung.
    
    Args:
        model_name: HuggingFace model name (z.B. "mistralai/Mistral-7B-v0.1")
        context_lengths: Liste von Context-Längen zum Testen
        quantize: True = INT8, False = FP16
        output_dir: Output-Verzeichnis für Ergebnisse
        device: 'cuda' or 'cpu'
        seed: Random seed für Reproduzierbarkeit
        
    Returns:
        Dictionary mit allen Metriken
    """
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"\n{'='*80}")
    print(f"Experiment: {model_name}")
    print(f"Quantization: {'INT8' if quantize else 'FP16'}")
    print(f"Context Lengths: {context_lengths}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"{'='*80}\n")
    
    # Initialize VRAM profiler
    profiler = VRAMProfiler(device_id=0)
    vram_before = profiler.get_vram_usage()
    print(f"VRAM before model load: {vram_before['used_gb']:.2f} GB")
    
    # Load model & tokenizer
    print(f"\nLoading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='auto',
        low_cpu_mem_usage=True
    )
    model.eval()
    
    vram_after_model = profiler.get_vram_usage()
    print(f"VRAM after model load: {vram_after_model['used_gb']:.2f} GB")
    print(f"Model VRAM: {vram_after_model['used_gb'] - vram_before['used_gb']:.2f} GB")
    
    # Setup quantization hook (if enabled)
    quantizer = SymmetricINT8Quantizer()
    hook = KVCacheQuantizationHook(quantizer, enable=quantize)
    
    # Register hook auf alle Attention-Layer
    # NOTE: Dies ist modell-spezifisch. Mistral hat 'self_attn' in jedem Layer.
    if quantize:
        for name, module in model.named_modules():
            if 'self_attn' in name:
                module.register_forward_hook(hook)
        print(f"Registered INT8 quantization hooks on attention layers.")
    
    # Results storage
    results = {
        'experiment_id': f"{model_name.split('/')[-1]}_{'int8' if quantize else 'fp16'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'config': {
            'model': model_name,
            'kv_precision': 'int8' if quantize else 'fp16',
            'device': device,
            'seed': seed
        },
        'measurements': []
    }
    
    # Sample text für Perplexity (WikiText-2 style)
    sample_text = """
    Large language models have revolutionized natural language processing by demonstrating 
    remarkable capabilities in understanding and generating human-like text. These models, 
    trained on vast amounts of data, can perform a wide range of tasks from translation to 
    question answering. However, their deployment faces significant challenges, particularly 
    regarding computational resources and memory requirements. The key-value cache, which stores 
    attention keys and values during inference, becomes a major bottleneck for long-context 
    applications. As sequence lengths increase to 16k, 32k, or even 128k tokens, the KV cache 
    can consume more memory than the model weights themselves. This motivates research into 
    quantization techniques that can compress the KV cache while maintaining model quality.
    """
    
    # Test über verschiedene Context-Längen
    for ctx_len in tqdm(context_lengths, desc="Testing context lengths"):
        print(f"\n--- Context Length: {ctx_len} ---")
        
        # Generate dummy input (für VRAM-Messung ohne echte Daten)
        input_text = sample_text * (ctx_len // 100 + 1)  # Repeat to reach ctx_len
        encodings = tokenizer(
            input_text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=ctx_len
        )
        input_ids = encodings['input_ids'].to(device)
        actual_seq_len = input_ids.shape[1]
        
        print(f"Input tokens: {actual_seq_len}")
        
        # Forward pass mit KV-Cache
        vram_before_fwd = profiler.get_vram_usage()
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)
        elapsed_time = time.time() - start_time
        
        vram_after_fwd = profiler.get_vram_usage()
        
        # KV-Cache size measurement
        kv_size = measure_kvcache_size(outputs.past_key_values)
        
        # Perplexity (auf kleinerem Sample für Speed)
        ppl_sample = sample_text[:500]  # Erste 500 chars
        perplexity = compute_perplexity(model, tokenizer, ppl_sample, device)
        
        # Store measurement
        measurement = {
            'context_length': actual_seq_len,
            'kv_cache': {
                'total_bytes': kv_size['total_bytes'],
                'total_gb': kv_size['total_gb'],
                'bytes_per_token': kv_size['bytes_per_token']
            },
            'vram': {
                'before_gb': vram_before_fwd['used_gb'],
                'after_gb': vram_after_fwd['used_gb'],
                'delta_gb': vram_after_fwd['used_gb'] - vram_before_fwd['used_gb']
            },
            'latency_ms': elapsed_time * 1000,
            'perplexity': perplexity,
            'timestamp': datetime.now().isoformat()
        }
        
        results['measurements'].append(measurement)
        
        print(f"KV-Cache: {kv_size['total_gb']:.4f} GB ({kv_size['bytes_per_token']:.1f} bytes/token)")
        print(f"VRAM Delta: {measurement['vram']['delta_gb']:.2f} GB")
        print(f"Latency: {measurement['latency_ms']:.1f} ms")
        print(f"Perplexity: {perplexity:.2f}")
        
        # Cleanup
        del outputs
        torch.cuda.empty_cache()
    
    # Save results
    output_file = output_dir / f"{results['experiment_id']}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    return results


# ============================================================================
# 5. Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='KV-Cache INT8 Quantization Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # FP16 Baseline
  python quantize_kvcache_int8.py --model mistralai/Mistral-7B-v0.1 --no-quantize
  
  # INT8 Quantization
  python quantize_kvcache_int8.py --model mistralai/Mistral-7B-v0.1 --quantize
  
  # Custom context lengths
  python quantize_kvcache_int8.py --model mistralai/Mistral-7B-v0.1 --quantize --context-lengths 512 1024 2048 4096
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='mistralai/Mistral-7B-v0.1',
        help='HuggingFace model name (default: Mistral-7B-v0.1)'
    )
    
    parser.add_argument(
        '--context-lengths',
        nargs='+',
        type=int,
        default=[512, 1024, 2048, 4096],
        help='Context lengths to test (default: 512 1024 2048 4096)'
    )
    
    parser.add_argument(
        '--quantize',
        action='store_true',
        help='Enable INT8 quantization (default: False = FP16 baseline)'
    )
    
    parser.add_argument(
        '--no-quantize',
        dest='quantize',
        action='store_false',
        help='Disable quantization (FP16 baseline)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('../results/raw'),
        help='Output directory for results (default: ../results/raw)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device (default: cuda)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Validate
    if not torch.cuda.is_available() and args.device == 'cuda':
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Create output dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiment
    try:
        results = run_experiment(
            model_name=args.model,
            context_lengths=args.context_lengths,
            quantize=args.quantize,
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed
        )
        
        print("\n" + "="*80)
        print("✅ Experiment completed successfully!")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
