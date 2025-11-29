#!/usr/bin/env python3
"""
KV-Cache Quantization Experiment Runner
========================================

Central orchestrator for running KV-cache quantization experiments across:
- Multiple models (GPT-2, Mistral, Qwen, etc.)
- Multiple backends (quanto, hqq)
- Multiple bit-widths (INT2, INT4, INT8)
- Multiple methods (baseline, KIVI, ATOM) [TODO]

Usage:
    # Interactive mode (menu-driven)
    python run_experiments.py --interactive
    
    # Run specific experiment
    python run_experiments.py --model gpt2 --method baseline --nbits 4 --backend hqq
    
    # Run full matrix (all combinations)
    python run_experiments.py --full-matrix --models gpt2 "Qwen/Qwen2-0.5B"
    
    # Quick validation run
    python run_experiments.py --quick-test

Author: Lennart Behr
Date: November 2025
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    model: str
    method: str  # 'baseline', 'kivi', 'atom'
    nbits: int
    backend: str  # 'quanto', 'hqq', 'custom'
    context_lengths: List[int]
    seed: int = 42
    device: str = 'cuda'
    output_dir: str = '../results/raw'


# Supported configurations
# 
# MODEL STRATEGY:
# - GPT-2 variants: ONLY for method validation (max 512 tokens safe, 1024 causes CUDA errors)
#   → No GQA, outdated architecture, NOT for final results!
# - Qwen2-0.5B: Quick long-context validation (GQA 7:1, 32k context)
# - Mistral-7B: Primary evaluation model for LongBench/NarrativeQA (GQA 4:1, 32k context)
#
SUPPORTED_MODELS = {
    # === METHOD VALIDATION ONLY (short context) ===
    'gpt2': {
        'vram_gb': 0.5, 
        'max_context': 512,  # Safe limit (1024 causes CUDA index errors)
        'description': 'GPT-2 Small (124M) - METHOD VALIDATION ONLY',
        'role': 'validation',
        'gqa_ratio': None,  # No GQA
    },
    'gpt2-medium': {
        'vram_gb': 1.5, 
        'max_context': 512, 
        'description': 'GPT-2 Medium (355M) - METHOD VALIDATION ONLY',
        'role': 'validation',
        'gqa_ratio': None,
    },
    # === LONG-CONTEXT EVALUATION ===
    'Qwen/Qwen2-0.5B': {
        'vram_gb': 1.0, 
        'max_context': 4096,  # Safe for testing, supports up to 32k
        'description': 'Qwen2 0.5B - Quick validation (WARNING: 2 KV-heads, INT4 fails!)',
        'role': 'evaluation',
        'gqa_ratio': '7:1',
        'kv_heads': 2,
    },
    'Qwen/Qwen2-7B': {
        'vram_gb': 14.0, 
        'max_context': 4096,  # Safe for testing, supports up to 128k
        'description': 'Qwen2 7B - Test if 4 KV-heads is sufficient for INT4',
        'role': 'evaluation',
        'gqa_ratio': '7:1',
        'kv_heads': 4,
    },
    'mistralai/Mistral-7B-v0.1': {
        'vram_gb': 14.0, 
        'max_context': 4096,  # Safe for testing, supports up to 32k
        'description': 'Mistral 7B - PRIMARY for LongBench/NarrativeQA',
        'role': 'evaluation',
        'gqa_ratio': '4:1',
        'kv_heads': 8,
    },
}

SUPPORTED_METHODS = {
    'baseline': {
        'description': 'Standard HuggingFace QuantizedCache (quanto/hqq)',
        'script': 'quantize_kvcache_hf.py',
        'backends': ['none', 'quanto', 'hqq'],  # 'none' = FP16 baseline
        'nbits': {'none': [16], 'quanto': [4], 'hqq': [4, 8]},  # Skip INT2 (too aggressive), quanto INT8 broken
    },
    'kivi': {
        'description': 'KIVI: Asymmetric quantization (2-bit keys, 4-bit values)',
        'script': 'quantize_kvcache_kivi.py',  # TODO: implement
        'backends': ['custom'],
        'nbits': {'custom': ['2k4v']},  # Special: 2-bit keys, 4-bit values
    },
    'atom': {
        'description': 'ATOM: Outlier-aware mixed-precision',
        'script': 'quantize_kvcache_atom.py',  # TODO: implement
        'backends': ['custom'],
        'nbits': {'custom': [4, 8]},
    },
}

# Default context lengths for different scenarios
CONTEXT_PRESETS = {
    'quick': [128, 256],
    'standard': [128, 256, 512, 1024],
    'long': [512, 1024, 2048, 4096],
    'full': [128, 256, 512, 1024, 2048, 4096, 8192],
}


# ============================================================================
# Experiment Runner
# ============================================================================

class ExperimentRunner:
    """Orchestrates KV-cache quantization experiments."""
    
    def __init__(self, scripts_dir: Path = None):
        self.scripts_dir = scripts_dir or Path(__file__).parent
        self.results = []
    
    def run_single(self, config: ExperimentConfig) -> Dict:
        """Run a single experiment configuration."""
        method_info = SUPPORTED_METHODS.get(config.method)
        if not method_info:
            raise ValueError(f"Unknown method: {config.method}")
        
        script = self.scripts_dir / method_info['script']
        if not script.exists():
            raise FileNotFoundError(f"Script not found: {script}")
        
        # Build command
        cmd = [
            sys.executable, str(script),
            '--model', config.model,
            '--context-lengths', *[str(c) for c in config.context_lengths],
            '--seed', str(config.seed),
            '--device', config.device,
            '--output-dir', config.output_dir,
        ]
        
        # Add quantization flags based on method
        if config.method == 'baseline':
            if config.nbits == 16:
                cmd.append('--no-quantize')
            else:
                cmd.extend(['--quantize', '--nbits', str(config.nbits), '--backend', config.backend])
        elif config.method in ['kivi', 'atom']:
            # Custom methods have their own handling
            cmd.extend(['--nbits', str(config.nbits)])
        
        print(f"\n{'='*60}")
        print(f"Running: {config.method} | {config.model} | {config.nbits}-bit | {config.backend}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}\n")
        
        # Run
        result = subprocess.run(cmd, capture_output=False)
        
        return {
            'config': config.__dict__,
            'returncode': result.returncode,
            'timestamp': datetime.now().isoformat(),
        }
    
    def run_matrix(self, 
                   models: List[str], 
                   methods: List[str], 
                   context_preset: str = 'standard') -> List[Dict]:
        """Run experiments for all combinations."""
        results = []
        context_lengths = CONTEXT_PRESETS.get(context_preset, CONTEXT_PRESETS['standard'])
        
        for model in models:
            # Check VRAM requirement
            model_info = SUPPORTED_MODELS.get(model, {})
            max_context = model_info.get('max_context', 1024)
            valid_contexts = [c for c in context_lengths if c <= max_context]
            
            for method in methods:
                method_info = SUPPORTED_METHODS[method]
                
                for backend in method_info['backends']:
                    for nbits in method_info['nbits'].get(backend, []):
                        config = ExperimentConfig(
                            model=model,
                            method=method,
                            nbits=nbits,
                            backend=backend,
                            context_lengths=valid_contexts,
                        )
                        
                        try:
                            result = self.run_single(config)
                            results.append(result)
                        except Exception as e:
                            print(f"❌ Failed: {e}")
                            results.append({
                                'config': config.__dict__,
                                'error': str(e),
                                'timestamp': datetime.now().isoformat(),
                            })
        
        self.results = results
        return results
    
    def quick_test(self) -> Dict:
        """Run a quick validation test."""
        config = ExperimentConfig(
            model='gpt2',
            method='baseline',
            nbits=4,
            backend='hqq',
            context_lengths=[128, 256],
        )
        return self.run_single(config)


# ============================================================================
# Interactive Mode
# ============================================================================

def interactive_mode():
    """Interactive menu for experiment configuration."""
    runner = ExperimentRunner()
    
    while True:
        print("\n" + "="*60)
        print("KV-Cache Quantization Experiment Runner")
        print("="*60)
        print("\n1. Quick Test (GPT-2, INT4-hqq, 128/256 tokens)")
        print("2. Run Single Experiment (configure)")
        print("3. Run Model Comparison (multiple models, same config)")
        print("4. Run Full Baseline Matrix (all baseline combinations)")
        print("5. Show Supported Configurations")
        print("6. Exit")
        
        choice = input("\nSelect option [1-6]: ").strip()
        
        if choice == '1':
            print("\n🚀 Running quick test...")
            runner.quick_test()
        
        elif choice == '2':
            config = configure_single_experiment()
            if config:
                runner.run_single(config)
        
        elif choice == '3':
            models = select_models()
            if models:
                print("\nUsing default config: INT4-hqq, standard context lengths")
                for model in models:
                    config = ExperimentConfig(
                        model=model,
                        method='baseline',
                        nbits=4,
                        backend='hqq',
                        context_lengths=CONTEXT_PRESETS['standard'],
                    )
                    runner.run_single(config)
        
        elif choice == '4':
            print("\n⚠️  This will run many experiments. Continue? [y/N]")
            if input().lower() == 'y':
                runner.run_matrix(
                    models=['gpt2'],
                    methods=['baseline'],
                    context_preset='standard'
                )
        
        elif choice == '5':
            show_supported_configs()
        
        elif choice == '6':
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid option.")


def configure_single_experiment() -> Optional[ExperimentConfig]:
    """Configure a single experiment interactively."""
    print("\n--- Configure Experiment ---\n")
    
    # Model
    print("Available models:")
    for i, (name, info) in enumerate(SUPPORTED_MODELS.items(), 1):
        print(f"  {i}. {name} ({info['description']}, ~{info['vram_gb']} GB)")
    model_idx = int(input("\nSelect model [1-5]: ").strip()) - 1
    model = list(SUPPORTED_MODELS.keys())[model_idx]
    
    # Method
    print("\nAvailable methods:")
    for i, (name, info) in enumerate(SUPPORTED_METHODS.items(), 1):
        status = "✅" if Path(f"scripts/{info['script']}").exists() or info['script'] == 'quantize_kvcache_hf.py' else "❌ TODO"
        print(f"  {i}. {name}: {info['description']} {status}")
    method_idx = int(input("\nSelect method [1-3]: ").strip()) - 1
    method = list(SUPPORTED_METHODS.keys())[method_idx]
    
    # Backend & nbits based on method
    method_info = SUPPORTED_METHODS[method]
    backend = method_info['backends'][0]
    if len(method_info['backends']) > 1:
        print(f"\nAvailable backends: {method_info['backends']}")
        backend = input("Select backend: ").strip()
    
    nbits_options = method_info['nbits'].get(backend, [4])
    print(f"\nAvailable bit-widths: {nbits_options}")
    nbits = int(input("Select nbits: ").strip()) if len(nbits_options) > 1 else nbits_options[0]
    
    # Context lengths
    print(f"\nContext presets: {list(CONTEXT_PRESETS.keys())}")
    preset = input("Select preset [quick/standard/long/full]: ").strip() or 'standard'
    context_lengths = CONTEXT_PRESETS.get(preset, CONTEXT_PRESETS['standard'])
    
    return ExperimentConfig(
        model=model,
        method=method,
        nbits=nbits,
        backend=backend,
        context_lengths=context_lengths,
    )


def select_models() -> List[str]:
    """Select multiple models interactively."""
    print("\nAvailable models:")
    models = list(SUPPORTED_MODELS.keys())
    for i, name in enumerate(models, 1):
        info = SUPPORTED_MODELS[name]
        print(f"  {i}. {name} (~{info['vram_gb']} GB)")
    
    selection = input("\nSelect models (comma-separated, e.g., 1,2,3): ").strip()
    indices = [int(i.strip()) - 1 for i in selection.split(',')]
    return [models[i] for i in indices if 0 <= i < len(models)]


def show_supported_configs():
    """Display all supported configurations."""
    print("\n" + "="*60)
    print("SUPPORTED MODELS")
    print("="*60)
    for name, info in SUPPORTED_MODELS.items():
        print(f"  {name}")
        print(f"    - VRAM: ~{info['vram_gb']} GB")
        print(f"    - Max context: {info['max_context']} tokens")
        print(f"    - {info['description']}")
    
    print("\n" + "="*60)
    print("SUPPORTED METHODS")
    print("="*60)
    for name, info in SUPPORTED_METHODS.items():
        script_exists = Path(f"{info['script']}").name == 'quantize_kvcache_hf.py'
        status = "✅ Implemented" if script_exists else "❌ TODO"
        print(f"  {name} ({status})")
        print(f"    - {info['description']}")
        print(f"    - Backends: {info['backends']}")
        print(f"    - Bit-widths: {info['nbits']}")
    
    print("\n" + "="*60)
    print("CONTEXT PRESETS")
    print("="*60)
    for name, lengths in CONTEXT_PRESETS.items():
        print(f"  {name}: {lengths}")


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="KV-Cache Quantization Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python run_experiments.py --interactive
  
  # Quick test
  python run_experiments.py --quick-test
  
  # Single experiment
  python run_experiments.py --model gpt2 --method baseline --nbits 4 --backend hqq
  
  # Full matrix on GPT-2
  python run_experiments.py --full-matrix --models gpt2
        """
    )
    
    # Mode selection
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--interactive', '-i', action='store_true',
                      help="Run in interactive mode (menu-driven)")
    mode.add_argument('--quick-test', '-q', action='store_true',
                      help="Run quick validation test")
    mode.add_argument('--full-matrix', action='store_true',
                      help="Run full experiment matrix")
    
    # Single experiment config
    parser.add_argument('--model', type=str, help="Model name or HuggingFace path")
    parser.add_argument('--method', type=str, default='baseline',
                        choices=['baseline', 'kivi', 'atom'],
                        help="Quantization method")
    parser.add_argument('--nbits', type=int, default=4, help="Quantization bits")
    parser.add_argument('--backend', type=str, default='hqq',
                        choices=['quanto', 'hqq', 'custom'],
                        help="Quantization backend")
    parser.add_argument('--context-preset', type=str, default='standard',
                        choices=['quick', 'standard', 'long', 'full'],
                        help="Context length preset")
    
    # Matrix config
    parser.add_argument('--models', type=str, nargs='+',
                        help="Models for matrix run")
    parser.add_argument('--methods', type=str, nargs='+', default=['baseline'],
                        help="Methods for matrix run")
    
    args = parser.parse_args()
    runner = ExperimentRunner()
    
    if args.interactive:
        interactive_mode()
    
    elif args.quick_test:
        runner.quick_test()
    
    elif args.full_matrix:
        models = args.models or ['gpt2']
        runner.run_matrix(models, args.methods, args.context_preset)
    
    elif args.model:
        config = ExperimentConfig(
            model=args.model,
            method=args.method,
            nbits=args.nbits,
            backend=args.backend,
            context_lengths=CONTEXT_PRESETS[args.context_preset],
        )
        runner.run_single(config)
    
    else:
        parser.print_help()
        print("\n💡 Tip: Use --interactive for menu-driven mode")


if __name__ == '__main__':
    main()
