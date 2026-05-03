#!/usr/bin/env python3
"""Smoke test for Gemma-4-E4B compatibility with profiler infrastructure.

Checks:
1. Model loads via AutoModelForImageTextToText
2. AutoTokenizer works for text-only input
3. Forward pass with just input_ids (no images) succeeds
4. DynamicCache captures KV tensors
5. Attention layer naming matches .self_attn pattern
6. model.config.text_config has expected architecture fields
7. KV-Sharing: which layers share KV entries

Usage:
    python smoke_test_gemma4.py [--device cuda]
"""

import argparse
import gc
import sys

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer, DynamicCache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-E4B")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Gemma-4-E4B Smoke Test")
    print(f"  Model:  {args.model}")
    print(f"  Device: {args.device}")
    print("=" * 70)

    # 1. Config check
    print("\n[1/7] Loading config...")
    config = AutoConfig.from_pretrained(args.model)
    print(f"  model_type: {config.model_type}")
    print(f"  has text_config: {hasattr(config, 'text_config')}")
    tc = config.text_config
    print(f"  text_config.model_type: {tc.model_type}")
    print(f"  num_hidden_layers: {tc.num_hidden_layers}")
    print(f"  num_attention_heads: {tc.num_attention_heads}")
    print(f"  num_key_value_heads: {tc.num_key_value_heads}")
    print(f"  head_dim: {tc.head_dim}")
    print(f"  hidden_size: {tc.hidden_size}")
    print(f"  num_kv_shared_layers: {getattr(tc, 'num_kv_shared_layers', 'N/A')}")
    print(f"  attention_k_eq_v: {getattr(tc, 'attention_k_eq_v', 'N/A')}")
    print(f"  sliding_window: {getattr(tc, 'sliding_window', 'N/A')}")
    layer_types = getattr(tc, "layer_types", [])
    n_full = sum(1 for lt in layer_types if lt == "full_attention")
    n_sliding = sum(1 for lt in layer_types if lt == "sliding_attention")
    print(f"  layer_types: {n_full} full + {n_sliding} sliding = {len(layer_types)} total")

    # 2. Tokenizer
    print("\n[2/7] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    test_text = "Hello, this is a smoke test for Gemma-4."
    tokens = tokenizer(test_text, return_tensors="pt")
    print(f"  Tokenizer class: {type(tokenizer).__name__}")
    print(f"  pad_token: {tokenizer.pad_token}")
    print(f"  eos_token: {tokenizer.eos_token}")
    print(f"  Test input: {tokens['input_ids'].shape}")

    # 3. Model loading
    print("\n[3/7] Loading model (this may take a while on first run)...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map=args.device,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model class: {type(model).__name__}")
    print(f"  Parameters: {num_params / 1e9:.2f}B")
    if args.device == "cuda":
        vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"  VRAM after load: {vram_mb:.0f} MB")

    # 4. Attention layer names
    print("\n[4/7] Checking attention layer naming...")
    attn_layers = []
    for name, module in model.named_modules():
        if name.endswith(".self_attn"):
            attn_layers.append(name)
    print(f"  Found {len(attn_layers)} layers matching '.self_attn'")
    if attn_layers:
        print(f"  First: {attn_layers[0]}")
        print(f"  Last:  {attn_layers[-1]}")
    else:
        print("  WARNING: No .self_attn layers found! Checking all module names...")
        for name, _ in model.named_modules():
            if "attn" in name.lower():
                print(f"    {name}")

    # 5. Forward pass (text-only)
    print("\n[5/7] Running text-only forward pass...")
    input_ids = tokens["input_ids"].to(args.device)
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    print(f"  Output type: {type(outputs).__name__}")
    print(f"  Has logits: {hasattr(outputs, 'logits')}")
    if hasattr(outputs, "logits"):
        print(f"  Logits shape: {outputs.logits.shape}")
    print(f"  Has past_key_values: {hasattr(outputs, 'past_key_values')}")

    # 6. DynamicCache capture
    print("\n[6/7] Testing DynamicCache KV capture...")
    cache = DynamicCache()
    with torch.no_grad():
        outputs = model(input_ids, past_key_values=cache, use_cache=True)
    kv_cache = outputs.past_key_values

    if hasattr(kv_cache, "layers"):
        print(f"  Cache type: {type(kv_cache).__name__}")
        print(f"  Cache layers: {len(kv_cache.layers)}")
        # Check first and last layer
        for i in [0, len(kv_cache.layers) - 1]:
            layer = kv_cache.layers[i]
            if hasattr(layer, "keys") and layer.keys is not None:
                print(f"  Layer {i}: key={layer.keys.shape}, value={layer.values.shape}")
            else:
                print(f"  Layer {i}: keys=None (shared layer?)")
    else:
        print(f"  Cache type: {type(kv_cache).__name__}")
        print(f"  WARNING: No .layers attribute — old cache API?")
        if hasattr(kv_cache, "key_cache"):
            print(f"  key_cache length: {len(kv_cache.key_cache)}")

    # 7. KV-Sharing analysis
    print("\n[7/7] Analyzing KV-Sharing...")
    if hasattr(kv_cache, "layers"):
        empty_layers = []
        populated_layers = []
        for i, layer in enumerate(kv_cache.layers):
            if hasattr(layer, "keys") and layer.keys is not None and layer.keys.numel() > 0:
                populated_layers.append(i)
            else:
                empty_layers.append(i)
        print(f"  Populated KV layers: {len(populated_layers)}")
        print(f"  Empty/shared KV layers: {len(empty_layers)}")
        if empty_layers:
            print(f"  Empty layer indices: {empty_layers}")

        # KV size estimation
        total_kv_bytes = 0
        for i in populated_layers[:1]:
            layer = kv_cache.layers[i]
            k_shape = layer.keys.shape
            v_shape = layer.values.shape
            bytes_per_element = layer.keys.element_size()
            kv_per_layer = (k_shape.numel() + v_shape.numel()) * bytes_per_element
            total_kv_bytes = kv_per_layer * len(populated_layers)
            print(f"  Key shape per layer: {list(k_shape)}")
            print(f"  Value shape per layer: {list(v_shape)}")
            print(f"  Estimated total KV for {len(populated_layers)} layers: {total_kv_bytes / 1024 / 1024:.2f} MB")

    # Summary
    print("\n" + "=" * 70)
    print("SMOKE TEST RESULTS:")
    checks = {
        "Config has text_config": hasattr(config, "text_config"),
        "Tokenizer works": tokens["input_ids"].shape[-1] > 0,
        "Model loads": num_params > 0,
        ".self_attn layers found": len(attn_layers) > 0,
        "Text-only forward OK": hasattr(outputs, "logits"),
        "DynamicCache works": hasattr(kv_cache, "layers"),
    }
    all_ok = True
    for check, ok in checks.items():
        status = "✓" if ok else "✗"
        print(f"  {status} {check}")
        if not ok:
            all_ok = False
    print("=" * 70)
    print(f"Result: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")

    del model, cache, kv_cache
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
