"""
nanoGPT Inference with KV-Cache Size Tracking

Generiert Text und misst VRAM-Verbrauch des KV-Cache bei verschiedenen Context-Lengths.
Berechnet theoretische vs. gemessene KV-Cache-Größe.

Usage:
    cd kv-cache-quantization/scripts/
    python infer_nanogpt_kvcache.py --max_new_tokens=256 --context_lengths=256,512,1024,2048
"""

import os
import sys
import time
import pickle

import numpy as np
import torch

# Add parent dir for profiler
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.profiler import VRAMProfiler

# Add nanoGPT to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'nanoGPT'))
from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
checkpoint_path = '../nanoGPT/out-shakespeare-char/ckpt.pt'
data_dir = '../nanoGPT/data/shakespeare_char'
device = 'cuda'
dtype = 'float16'
seed = 42
temperature = 0.8
top_k = 200
max_new_tokens = 256
context_lengths = [128, 256, 512, 1024]  # Different starting contexts to measure

# Output
vram_log_path = '../results/raw/nanogpt_inference_vram.json'

# Override from command line
for arg in sys.argv[1:]:
    if arg.startswith('--'):
        key, val = arg[2:].split('=')
        if key == 'context_lengths':
            context_lengths = [int(x) for x in val.split(',')]
        elif key in globals():
            val_type = type(globals()[key])
            globals()[key] = val_type(val)
            print(f"Overriding: {key} = {val}")

# -----------------------------------------------------------------------------
# Initialize
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("🔬 Initializing VRAM Profiler")
print("="*60)
profiler = VRAMProfiler()
profiler.log_vram("Baseline (before model load)")

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Load metadata
meta_path = os.path.join(data_dir, 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

stoi = meta['stoi']
itos = meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(f"Vocab size: {len(itos)}")

# -----------------------------------------------------------------------------
# Load Model
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("🧠 Loading Model from Checkpoint")
print("="*60)

checkpoint = torch.load(checkpoint_path, map_location=device)
model_args = checkpoint['model_args']
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)

state_dict = checkpoint['model']
# Fix checkpoint keys if needed
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.to(device)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params/1e6:.2f}M")
print(f"Config: {model_args}")

profiler.log_vram("Model loaded")

# -----------------------------------------------------------------------------
# Helper: Calculate Theoretical KV-Cache Size
# -----------------------------------------------------------------------------
def calculate_kv_cache_size(n_layer, batch_size, seq_len, n_embd, dtype='fp16'):
    """
    KV-Cache Size = n_layer × 2 (Key + Value) × batch × seq_len × n_embd × bytes_per_element
    
    Args:
        n_layer: Number of transformer layers
        batch_size: Batch size (usually 1 for inference)
        seq_len: Sequence length (context)
        n_embd: Embedding dimensions
        dtype: Data type ('fp16' = 2 bytes, 'int8' = 1 byte)
    
    Returns:
        Size in bytes
    """
    bytes_per_element = 2 if dtype == 'fp16' else 1
    
    # Each layer stores Key and Value (2×)
    # Each Key/Value: [batch, n_head, seq_len, head_dim]
    # But stored as [batch, seq_len, n_embd] before splitting
    size_bytes = n_layer * 2 * batch_size * seq_len * n_embd * bytes_per_element
    
    return size_bytes

# -----------------------------------------------------------------------------
# Inference Loop with Context-Length Sweep
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("🚀 Running Inference with KV-Cache Tracking")
print("="*60)

# Starting prompt (use same for all runs)
start = "ROMEO:"
start_ids = encode(start)

results = []

for context_len in context_lengths:
    print(f"\n{'='*60}")
    print(f"📊 Context Length: {context_len} tokens")
    print(f"{'='*60}")
    
    # Prepare context (pad or truncate to context_len)
    if len(start_ids) >= context_len:
        context_ids = start_ids[:context_len]
    else:
        # Pad with zeros (will be embedded to something)
        context_ids = start_ids + [0] * (context_len - len(start_ids))
    
    x = torch.tensor(context_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, context_len)
    
    print(f"Context: '{decode(context_ids[:50])}...' ({len(context_ids)} tokens)")
    
    # Measure VRAM before generation
    profiler.log_vram(f"Before generation (context={context_len})")
    vram_before = profiler.measure_vram_mb()
    
    # Generate tokens (this will store activations in memory)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass
            logits, _ = model(x)
            
            # Take last token's logits
            logits = logits[:, -1, :]  # (B, vocab_size)
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            x = torch.cat([x, next_token], dim=1)  # Growing context!
            
            # Optional: Truncate to max block_size
            if x.size(1) > model.config.block_size:
                x = x[:, -model.config.block_size:]
    
    t1 = time.time()
    elapsed = t1 - t0
    
    # Measure VRAM after generation
    profiler.log_vram(f"After generation (context={context_len}, generated={max_new_tokens})")
    vram_after = profiler.measure_vram_mb()
    
    # Calculate VRAM delta (this includes activations + KV-cache approximation)
    vram_delta = vram_after - vram_before
    
    # Calculate theoretical KV-Cache size
    final_seq_len = min(x.size(1), model.config.block_size)
    theoretical_kv_mb = calculate_kv_cache_size(
        n_layer=model_args['n_layer'],
        batch_size=1,
        seq_len=final_seq_len,
        n_embd=model_args['n_embd'],
        dtype='fp16'
    ) / 1e6
    
    # Decode generated text
    generated_ids = x[0].tolist()
    generated_text = decode(generated_ids[context_len:context_len+50])  # First 50 chars
    
    print(f"\n📝 Generated (first 50 chars): '{generated_text}...'")
    print(f"⏱️  Generation time: {elapsed:.2f}s ({max_new_tokens/elapsed:.1f} tokens/s)")
    print(f"📊 VRAM Delta: {vram_delta:.1f} MB")
    print(f"📐 Theoretical KV-Cache: {theoretical_kv_mb:.1f} MB")
    print(f"📏 Final sequence length: {final_seq_len} tokens")
    
    # Note: VRAM Delta is NOT just KV-Cache!
    # It includes: KV-Cache + Activations + Temporary buffers
    # For true KV-Cache isolation, we'd need to modify model.py to return past_key_values
    print(f"⚠️  Note: VRAM Delta includes KV-Cache + Activations (not isolated)")
    
    results.append({
        'context_len': context_len,
        'max_new_tokens': max_new_tokens,
        'final_seq_len': final_seq_len,
        'vram_before_mb': vram_before,
        'vram_after_mb': vram_after,
        'vram_delta_mb': vram_delta,
        'theoretical_kv_mb': theoretical_kv_mb,
        'generation_time_s': elapsed,
        'tokens_per_sec': max_new_tokens / elapsed,
        'generated_text_preview': generated_text
    })

# -----------------------------------------------------------------------------
# Save Results
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("💾 Saving VRAM Log")
print("="*60)

os.makedirs(os.path.dirname(vram_log_path), exist_ok=True)
profiler.save_to_json(vram_log_path)

# Save additional results
import json
results_path = vram_log_path.replace('.json', '_results.json')
with open(results_path, 'w') as f:
    json.dump({
        'model_args': model_args,
        'inference_config': {
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'top_k': top_k,
        },
        'results': results
    }, f, indent=2)

print(f"✅ VRAM log saved to: {vram_log_path}")
print(f"✅ Results saved to: {results_path}")
print(f"📊 Peak VRAM Delta: {profiler.get_peak_vram_mb():.1f} MB")

# -----------------------------------------------------------------------------
# Summary Table
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("📊 Summary: KV-Cache Scaling")
print("="*60)
print(f"{'Context':<10} {'Final Seq':<12} {'VRAM Δ':<12} {'Theoretical KV':<18} {'Tokens/s':<10}")
print("-" * 60)
for r in results:
    print(f"{r['context_len']:<10} {r['final_seq_len']:<12} {r['vram_delta_mb']:>8.1f} MB  {r['theoretical_kv_mb']:>12.1f} MB     {r['tokens_per_sec']:>8.1f}")

print("\n" + "="*60)
print("⚠️  Important Note:")
print("="*60)
print("nanoGPT does NOT use KV-Cache by default (recomputes every token).")
print("VRAM Delta includes Activations, not isolated KV-Cache.")
print("For true KV-Cache measurement, model.py needs modification.")
print("\n→ Next step: Implement explicit KV-Cache in model.py")
print("="*60)

print("\n✅ Inference Complete!")
