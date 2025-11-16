"""
nanoGPT Training mit VRAM-Profiling

Vereinfachte Version von train.py mit VRAMProfiler-Integration.
Misst VRAM bei jedem Evaluation-Interval für wissenschaftliche Analyse.

Usage:
    cd kv-cache-quantization/scripts/
    python train_nanogpt_profiled.py --max_iters=500 --eval_interval=100
"""

import os
import sys
import time
import math
import pickle

import numpy as np
import torch

# Add parent dir to path for profiler import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.profiler import VRAMProfiler

# Add nanoGPT to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'nanoGPT'))
from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# Config (simplified for Shakespeare training)
# -----------------------------------------------------------------------------
out_dir = '../nanoGPT/out-shakespeare-char'
eval_interval = 100
eval_iters = 20
log_interval = 10

# Data
dataset = 'shakespeare_char'
data_dir = '../nanoGPT/data/shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

# Model (Baby GPT)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# Training
learning_rate = 1e-3
max_iters = 500
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

# System
device = 'cuda'
dtype = 'float16'
compile = False  # Disabled for clarity

# Profiling output
vram_log_path = '../results/raw/nanogpt_training_vram.json'

# Override from command line
import sys
for arg in sys.argv[1:]:
    if arg.startswith('--'):
        key, val = arg[2:].split('=')
        if key in globals():
            val_type = type(globals()[key])
            globals()[key] = val_type(val)
            print(f"Overriding: {key} = {val}")

# -----------------------------------------------------------------------------
# Initialize Profiler
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("🔬 Initializing VRAM Profiler")
print("="*60)
profiler = VRAMProfiler()
profiler.log_vram("Baseline (before model load)")

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("📊 Loading Data")
print("="*60)

def get_batch(split):
    """Load a batch of data from disk"""
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Load vocab size from metadata
meta_path = os.path.join(data_dir, 'meta.pkl')
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']
    print(f"Vocab size: {vocab_size}")
else:
    vocab_size = 65  # Default for character-level
    print(f"No meta.pkl found, defaulting to vocab_size={vocab_size}")

# -----------------------------------------------------------------------------
# Model Initialization
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("🧠 Creating Model")
print("="*60)

model_args = dict(
    n_layer=n_layer,
    n_head=n_head, 
    n_embd=n_embd,
    block_size=block_size,
    dropout=dropout,
    vocab_size=vocab_size,
    bias=False
)

gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)

# Count parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {n_params/1e6:.2f}M")

profiler.log_vram("Model loaded (FP16)")

# -----------------------------------------------------------------------------
# Optimizer
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("⚙️  Creating Optimizer")
print("="*60)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(beta1, beta2),
    weight_decay=weight_decay
)

profiler.log_vram("Optimizer created (AdamW)")

# -----------------------------------------------------------------------------
# Loss Estimation
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    """Evaluate on train and val splits"""
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("🚀 Starting Training")
print("="*60)

model.train()
X, Y = get_batch('train')  # Fetch first batch
t0 = time.time()
best_val_loss = float('inf')

for iter_num in range(max_iters):
    # Forward pass
    logits, loss = model(X, Y)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    
    # Fetch next batch while GPU is working
    X, Y = get_batch('train')
    
    # Logging
    if iter_num % log_interval == 0:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        print(f"Iter {iter_num:4d} | loss {loss.item():.4f} | time {dt*1000:.2f}ms")
    
    # Evaluation
    if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
        losses = estimate_loss()
        train_loss = losses['train'].item()
        val_loss = losses['val'].item()
        
        print(f"\n{'='*60}")
        print(f"📊 Evaluation @ Iter {iter_num}")
        print(f"{'='*60}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        
        # VRAM Profiling
        profiler.log_vram(f"Iter {iter_num} (train={train_loss:.4f}, val={val_loss:.4f})")
        
        # Save checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"✅ New best val_loss: {val_loss:.4f}")
            
            # Save checkpoint
            os.makedirs(out_dir, exist_ok=True)
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
            }
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
            print(f"💾 Checkpoint saved to {out_dir}/ckpt.pt")
        
        print(f"{'='*60}\n")

# -----------------------------------------------------------------------------
# Save VRAM Log
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("💾 Saving VRAM Log")
print("="*60)

os.makedirs(os.path.dirname(vram_log_path), exist_ok=True)
profiler.save_to_json(vram_log_path)
print(f"✅ VRAM log saved to: {vram_log_path}")
print(f"📊 Peak VRAM Delta: {profiler.get_peak_vram_mb():.1f} MB")

print("\n" + "="*60)
print("✅ Training Complete!")
print("="*60)
