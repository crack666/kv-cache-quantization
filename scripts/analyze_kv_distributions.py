#!/usr/bin/env python3
"""Analyze KV-cache tensor distributions across layers.

Captures Key and Value tensors via forward hooks during a single WikiText-2
forward pass, then computes per-layer statistics that predict quantization
tolerance: kurtosis, outlier ratios, dynamic range, variance ratios.

Usage:
    python analyze_kv_distributions.py \
        --model mistralai/Mistral-7B-v0.1 \
        --max-tokens 4096 \
        --seed 42 \
        --output-dir ../results/raw/kv_distributions/
"""

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ═══════════════════════════════════════════════════════════════════════════
# KV Tensor Capture via Hooks
# ═══════════════════════════════════════════════════════════════════════════

class KVCaptureHook:
    """Register forward hooks on attention layers to capture K and V tensors."""

    def __init__(self, model):
        self.kv_data = {}  # {layer_idx: {"key": Tensor, "value": Tensor}}
        self._hooks = []
        self._register(model)

    def _register(self, model):
        """Find attention layers and register hooks."""
        for name, module in model.named_modules():
            # Match common attention module names across architectures
            # Mistral/Llama: model.layers.X.self_attn
            # Qwen2: model.layers.X.self_attn
            if name.endswith(".self_attn"):
                layer_idx = int(name.split(".")[-2])
                hook = module.register_forward_hook(
                    self._make_hook(layer_idx)
                )
                self._hooks.append(hook)

        if not self._hooks:
            raise RuntimeError(
                "No attention layers found. Supported architectures: "
                "Mistral, Llama, Qwen2, Yi"
            )
        print(f"  Registered {len(self._hooks)} attention hooks")

    def _make_hook(self, layer_idx: int):
        """Create a hook that captures the K/V projections after the attention module."""
        def hook_fn(module, input, output):
            # output is typically (attn_output, attn_weights, past_key_value)
            # or (attn_output, past_key_value) depending on config
            # The past_key_value contains (key, value) for this layer
            # We access them from the module's internal computation

            # Strategy: Use k_proj and v_proj weights to re-derive K/V from hidden_states
            # But simpler: many attention modules store the projected tensors
            # Most reliable: capture from past_key_values in the output

            if isinstance(output, tuple):
                # Find the cache object in output
                for item in output:
                    if hasattr(item, "layers"):
                        try:
                            if len(item.layers) > layer_idx:
                                layer = item.layers[layer_idx]
                                self.kv_data[layer_idx] = {
                                    "key": layer.keys.detach().cpu().float(),
                                    "value": layer.values.detach().cpu().float(),
                                }
                        except (IndexError, AttributeError):
                            pass
        return hook_fn

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def capture_kv_from_cache(model, input_ids, device):
    """Alternative capture: run forward pass with DynamicCache and extract K/V directly."""
    from transformers import DynamicCache

    cache = DynamicCache()
    with torch.no_grad():
        model(input_ids, past_key_values=cache, use_cache=True)

    kv_data = {}
    for layer_idx, layer in enumerate(cache.layers):
        kv_data[layer_idx] = {
            "key": layer.keys.detach().cpu().float(),
            "value": layer.values.detach().cpu().float(),
        }

    del cache
    return kv_data


# ═══════════════════════════════════════════════════════════════════════════
# Statistical Analysis
# ═══════════════════════════════════════════════════════════════════════════

def compute_tensor_stats(tensor: torch.Tensor) -> dict:
    """Compute distribution statistics for a KV tensor.

    Args:
        tensor: Shape [batch, heads, seq_len, head_dim]

    Returns:
        dict with kurtosis, outlier ratios, dynamic range, variance ratios.
    """
    t = tensor.flatten().float()

    mean = t.mean().item()
    std = t.std().item()
    abs_t = t.abs()

    # Kurtosis (excess kurtosis: normal = 0)
    if std > 0:
        kurtosis = ((t - mean) / std).pow(4).mean().item() - 3.0
    else:
        kurtosis = 0.0

    # Outlier ratios
    n = t.numel()
    outlier_3sigma = (abs_t > 3 * std).sum().item() / n if std > 0 else 0.0
    outlier_6sigma = (abs_t > 6 * std).sum().item() / n if std > 0 else 0.0

    # Dynamic range: max(|x|) / mean(|x|)
    abs_mean = abs_t.mean().item()
    dynamic_range = abs_t.max().item() / abs_mean if abs_mean > 0 else 0.0

    # Per-channel vs per-token variance
    # tensor shape: [batch, heads, seq_len, head_dim]
    if tensor.dim() == 4:
        # Per-channel: variance across seq_len dimension (axis=2)
        per_channel_var = tensor.var(dim=2).mean().item()
        # Per-token: variance across head_dim dimension (axis=3)
        per_token_var = tensor.var(dim=3).mean().item()
        variance_ratio = per_channel_var / per_token_var if per_token_var > 0 else 0.0
    else:
        per_channel_var = 0.0
        per_token_var = 0.0
        variance_ratio = 0.0

    return {
        "mean": round(mean, 6),
        "std": round(std, 6),
        "min": round(t.min().item(), 6),
        "max": round(t.max().item(), 6),
        "kurtosis": round(kurtosis, 4),
        "outlier_ratio_3sigma": round(outlier_3sigma, 6),
        "outlier_ratio_6sigma": round(outlier_6sigma, 6),
        "dynamic_range": round(dynamic_range, 4),
        "per_channel_var": round(per_channel_var, 6),
        "per_token_var": round(per_token_var, 6),
        "variance_ratio": round(variance_ratio, 4),
        "numel": n,
    }


def analyze_layer(layer_idx: int, kv: dict) -> dict:
    """Analyze K and V tensors for a single layer."""
    k_stats = compute_tensor_stats(kv["key"])
    v_stats = compute_tensor_stats(kv["value"])

    return {
        "layer": layer_idx,
        "key": k_stats,
        "value": v_stats,
        "key_shape": list(kv["key"].shape),
        "value_shape": list(kv["value"].shape),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(description="Analyze KV-cache tensor distributions")
    p.add_argument("--model", required=True, help="HuggingFace model ID")
    p.add_argument("--max-tokens", type=int, default=4096, help="Max tokens from WikiText-2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default="../results/raw/kv_distributions/")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    t0 = time.time()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 80)
    print(f"KV Distribution Analysis")
    print(f"  Model:      {args.model}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Seed:       {args.seed}")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device,
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"  Loaded: {args.model} ({sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params)")

    # Load WikiText-2 input
    print("\nPreparing input from WikiText-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_tokens)
    input_ids = tokens["input_ids"].to(args.device)
    actual_len = input_ids.shape[-1]
    print(f"  Input tokens: {actual_len}")

    # Capture KV tensors via DynamicCache (most reliable method)
    print("\nRunning forward pass to capture KV tensors...")
    kv_data = capture_kv_from_cache(model, input_ids, args.device)
    print(f"  Captured {len(kv_data)} layers")

    # Analyze each layer
    print("\nAnalyzing distributions...")
    layer_stats = []
    for layer_idx in sorted(kv_data.keys()):
        stats = analyze_layer(layer_idx, kv_data[layer_idx])
        layer_stats.append(stats)

        # Print compact summary
        k = stats["key"]
        v = stats["value"]
        flag_k = " ⚠" if k["kurtosis"] > 3 else ""
        flag_v = " ⚠" if v["kurtosis"] > 3 else ""
        print(f"  Layer {layer_idx:>2}: K kurtosis={k['kurtosis']:>8.2f}{flag_k}  "
              f"V kurtosis={v['kurtosis']:>8.2f}{flag_v}  "
              f"K outlier_6σ={k['outlier_ratio_6sigma']:.4%}  "
              f"V outlier_6σ={v['outlier_ratio_6sigma']:.4%}")

    # Free GPU memory
    del kv_data, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Aggregate model-level summary
    k_kurtosis_all = [s["key"]["kurtosis"] for s in layer_stats]
    v_kurtosis_all = [s["value"]["kurtosis"] for s in layer_stats]
    k_outlier6_all = [s["key"]["outlier_ratio_6sigma"] for s in layer_stats]
    v_outlier6_all = [s["value"]["outlier_ratio_6sigma"] for s in layer_stats]

    summary = {
        "key_kurtosis_mean": round(np.mean(k_kurtosis_all), 4),
        "key_kurtosis_max": round(np.max(k_kurtosis_all), 4),
        "value_kurtosis_mean": round(np.mean(v_kurtosis_all), 4),
        "value_kurtosis_max": round(np.max(v_kurtosis_all), 4),
        "key_outlier_6sigma_mean": round(np.mean(k_outlier6_all), 6),
        "value_outlier_6sigma_mean": round(np.mean(v_outlier6_all), 6),
        "layers_with_heavy_tails": sum(1 for k in k_kurtosis_all if k > 3) + sum(1 for v in v_kurtosis_all if v > 3),
        "total_layers": len(layer_stats),
    }

    elapsed = time.time() - t0

    # Print summary
    print(f"\n{'='*80}")
    print(f"MODEL SUMMARY — {args.model}")
    print(f"{'='*80}")
    print(f"  Key kurtosis:   mean={summary['key_kurtosis_mean']:.2f}  max={summary['key_kurtosis_max']:.2f}")
    print(f"  Value kurtosis: mean={summary['value_kurtosis_mean']:.2f}  max={summary['value_kurtosis_max']:.2f}")
    print(f"  Key outlier 6σ:   {summary['key_outlier_6sigma_mean']:.4%}")
    print(f"  Value outlier 6σ: {summary['value_outlier_6sigma_mean']:.4%}")
    print(f"  Heavy-tail layers (kurtosis>3): {summary['layers_with_heavy_tails']}/{summary['total_layers']*2}")
    print(f"  Runtime: {elapsed:.1f}s")
    print(f"{'='*80}")

    # Save results
    result = {
        "model": args.model,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "max_tokens": actual_len,
            "seed": args.seed,
            "dataset": "wikitext-2-raw-v1",
        },
        "summary": summary,
        "layers": layer_stats,
        "runtime_s": round(elapsed, 1),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_short = args.model.split("/")[-1].lower().replace("-", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"kv_dist_{model_short}_{ts}.json"

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
