"""VRAM & Throughput Profiler with Baseline Subtraction.

Measures GPU memory usage via NVML (NVIDIA Management Library) with automatic
baseline subtraction to isolate experiment VRAM from Windows background processes.

Metrics:
- VRAM: Baseline-subtracted memory usage (MB)
- Throughput: Tokens/s, TTFT, TPT
- Peak Memory: torch.cuda.max_memory_allocated() (zero overhead)
- Power: Average watts during generation (background sampling)

Usage:
    profiler = VRAMProfiler()
    model = load_model().to('cuda')
    profiler.log_vram("Model Loaded")
    outputs = model(inputs)
    profiler.log_vram("After Forward")
    profiler.save_to_json("results/raw/vram_log.json")
"""

import pynvml
import json
import time
import threading
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, asdict, field


@dataclass
class ThroughputResult:
    """Container for throughput measurement results."""
    ttft_ms: float  # Time-to-First-Token in milliseconds
    tpt_ms: float   # Time-per-Token (average) in milliseconds
    tokens_per_second: float
    total_tokens: int
    total_time_ms: float
    prompt_tokens: int = 0
    generated_tokens: int = 0
    # New metrics
    peak_memory_mb: float = 0.0  # Peak GPU memory during generation
    avg_power_watts: float = 0.0  # Average power consumption
    energy_per_token_mj: float = 0.0  # Millijoules per token
    
    def to_dict(self) -> Dict:
        return asdict(self)


class PowerSampler:
    """Background thread for sampling GPU power with minimal overhead.
    
    Uses NVML nvmlDeviceGetPowerUsage() which is fast (~0.1ms).
    Samples every 50ms to balance accuracy vs overhead.
    """
    
    def __init__(self, handle, sample_interval_ms: int = 50):
        self.handle = handle
        self.sample_interval = sample_interval_ms / 1000.0
        self.samples: List[float] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
    
    def _sample_loop(self):
        """Background sampling loop."""
        while not self._stop_event.is_set():
            try:
                # nvmlDeviceGetPowerUsage returns milliwatts
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                self.samples.append(power_mw / 1000.0)  # Convert to Watts
            except pynvml.NVMLError:
                pass  # Ignore transient errors
            time.sleep(self.sample_interval)
    
    def start(self):
        """Start background power sampling."""
        self.samples = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> Dict[str, float]:
        """Stop sampling and return statistics."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        
        if not self.samples:
            return {"avg_watts": 0.0, "max_watts": 0.0, "min_watts": 0.0, "samples": 0}
        
        return {
            "avg_watts": sum(self.samples) / len(self.samples),
            "max_watts": max(self.samples),
            "min_watts": min(self.samples),
            "samples": len(self.samples)
        }


class VRAMProfiler:
    """Measures GPU VRAM with baseline subtraction for scientific measurements."""
    
    def __init__(self, gpu_index: int = 0):
        """Initialize NVML and measure baseline VRAM.
        
        Args:
            gpu_index: GPU device index (default: 0 for first GPU)
        """
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.baseline_mb = self._measure_raw()
        self.measurements: List[Dict] = []
        self.throughput_measurements: List[Dict] = []
        self.start_time = time.time()
        
        device_name = pynvml.nvmlDeviceGetName(self.handle)
        print(f"✅ VRAMProfiler initialized")
        print(f"   GPU: {device_name}")
        print(f"   Baseline: {self.baseline_mb:.1f} MB")
    
    def _measure_raw(self) -> float:
        """Measure raw VRAM (includes Windows/background processes).
        
        Returns:
            VRAM usage in megabytes (SI units: 10^6)
        """
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return info.used / 1e6
    
    def measure_vram_mb(self) -> float:
        """Measure VRAM delta (experiment only, baseline subtracted).
        
        Returns:
            VRAM delta in megabytes
        """
        return self._measure_raw() - self.baseline_mb
    
    def log_vram(self, label: str):
        """Log VRAM measurement with timestamp and label.
        
        Args:
            label: Description of current experiment state (e.g., "Model Loaded")
        """
        delta = self.measure_vram_mb()
        raw = self._measure_raw()
        elapsed = time.time() - self.start_time
        
        measurement = {
            "label": label,
            "timestamp_sec": round(elapsed, 3),
            "vram_delta_mb": round(delta, 2),
            "vram_raw_mb": round(raw, 2),
            "vram_baseline_mb": round(self.baseline_mb, 2)
        }
        
        self.measurements.append(measurement)
        
        print(f"📊 [{label}] Delta: {delta:.1f} MB | Raw: {raw:.1f} MB | t={elapsed:.1f}s")
    
    def get_peak_vram_mb(self) -> float:
        """Get peak VRAM delta during experiment.
        
        Returns:
            Maximum VRAM delta in megabytes
        """
        if not self.measurements:
            return 0.0
        return max(m["vram_delta_mb"] for m in self.measurements)
    
    def save_to_json(self, filepath: str):
        """Save measurements to JSON file.
        
        Args:
            filepath: Output JSON file path
        """
        output = {
            "gpu": pynvml.nvmlDeviceGetName(self.handle),
            "baseline_mb": round(self.baseline_mb, 2),
            "peak_vram_delta_mb": round(self.get_peak_vram_mb(), 2),
            "measurements": self.measurements,
            "throughput_measurements": self.throughput_measurements
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"💾 Saved to {filepath}")
        print(f"   Peak VRAM: {output['peak_vram_delta_mb']:.1f} MB")
        if self.throughput_measurements:
            print(f"   Throughput entries: {len(self.throughput_measurements)}")

    # =========================================================================
    # Throughput Measurement Methods
    # =========================================================================
    
    def measure_generation_throughput(
        self,
        generate_fn: Callable[[], Any],
        prompt_tokens: int,
        max_new_tokens: int,
        label: str = "generation",
        warmup_runs: int = 1,
        measure_power: bool = True
    ) -> ThroughputResult:
        """Measure throughput of a generation function.
        
        This measures:
        - TTFT: Time from start to first token (estimated)
        - TPT: Average time per subsequent token
        - Tokens/s: Overall throughput
        - Peak Memory: Maximum GPU memory allocated (zero overhead)
        - Power: Average watts during generation (background sampling)
        
        Args:
            generate_fn: Function that performs generation and returns output_ids
            prompt_tokens: Number of tokens in the prompt
            max_new_tokens: Maximum tokens to generate
            label: Label for this measurement
            warmup_runs: Number of warmup runs before measurement
            measure_power: Whether to sample power consumption (adds ~1% overhead)
            
        Returns:
            ThroughputResult with all timing and resource metrics
        """
        import torch
        
        # Warmup runs (not measured)
        for _ in range(warmup_runs):
            _ = generate_fn()
            torch.cuda.synchronize()
        
        # Clear cache and reset peak memory tracker
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        # Start power sampling (background thread)
        power_sampler = None
        if measure_power:
            power_sampler = PowerSampler(self.handle, sample_interval_ms=50)
            power_sampler.start()
        
        # Actual measurement
        start = time.perf_counter()
        torch.cuda.synchronize()
        
        output = generate_fn()
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        # Stop power sampling
        power_stats = {"avg_watts": 0.0, "samples": 0}
        if power_sampler:
            power_stats = power_sampler.stop()
        
        # Get peak memory (zero overhead - just reads counter)
        peak_memory_bytes = torch.cuda.max_memory_allocated()
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        
        # Calculate tokens generated
        if hasattr(output, 'shape'):
            total_output_tokens = output.shape[-1]
        elif hasattr(output, '__len__'):
            total_output_tokens = len(output[0]) if isinstance(output[0], (list, tuple)) else len(output)
        else:
            total_output_tokens = prompt_tokens + max_new_tokens
            
        generated_tokens = total_output_tokens - prompt_tokens
        total_time_ms = (end - start) * 1000
        total_time_s = total_time_ms / 1000
        
        # TTFT approximation: Time for first token ≈ total_time / (generated_tokens + prefill_overhead)
        # For accurate TTFT, use streaming callback (see measure_streaming_throughput)
        # Here we estimate: prefill is ~1 "token-equivalent" of time
        if generated_tokens > 0:
            tpt_ms = total_time_ms / generated_tokens
            ttft_ms = tpt_ms * 1.5  # Rough estimate: first token takes ~1.5x average
            tokens_per_second = generated_tokens / total_time_s
        else:
            tpt_ms = 0.0
            ttft_ms = total_time_ms
            tokens_per_second = 0.0
        
        # Calculate energy per token
        # Energy (Joules) = Power (Watts) × Time (seconds)
        # Energy per token (mJ) = (avg_watts × total_time_s × 1000) / generated_tokens
        if generated_tokens > 0 and power_stats["avg_watts"] > 0:
            total_energy_j = power_stats["avg_watts"] * total_time_s
            energy_per_token_mj = (total_energy_j * 1000) / generated_tokens
        else:
            energy_per_token_mj = 0.0
        
        result = ThroughputResult(
            ttft_ms=round(ttft_ms, 3),
            tpt_ms=round(tpt_ms, 3),
            tokens_per_second=round(tokens_per_second, 2),
            total_tokens=total_output_tokens,
            total_time_ms=round(total_time_ms, 3),
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            peak_memory_mb=round(peak_memory_mb, 1),
            avg_power_watts=round(power_stats["avg_watts"], 1),
            energy_per_token_mj=round(energy_per_token_mj, 2)
        )
        
        # Store measurement
        self.throughput_measurements.append({
            "label": label,
            "timestamp_sec": round(time.time() - self.start_time, 3),
            **result.to_dict(),
            "power_samples": power_stats.get("samples", 0)
        })
        
        # Enhanced output
        power_str = f" | ⚡{power_stats['avg_watts']:.0f}W" if power_stats["avg_watts"] > 0 else ""
        energy_str = f" | {energy_per_token_mj:.1f}mJ/tok" if energy_per_token_mj > 0 else ""
        print(f"⏱️  [{label}] {tokens_per_second:.1f} tok/s | TTFT≈{ttft_ms:.1f}ms | TPT={tpt_ms:.1f}ms | Peak={peak_memory_mb:.0f}MB{power_str}{energy_str}")
        
        return result
    
    def measure_streaming_throughput(
        self,
        model,
        input_ids,
        generation_config: Dict,
        label: str = "streaming"
    ) -> ThroughputResult:
        """Measure throughput with accurate TTFT using HuggingFace streamer.
        
        This provides accurate TTFT measurement by capturing exact timing
        of first token generation.
        
        Args:
            model: HuggingFace model
            input_ids: Input tensor (batch_size, seq_len)
            generation_config: Dict with generation parameters
            label: Label for this measurement
            
        Returns:
            ThroughputResult with accurate TTFT
        """
        import torch
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        prompt_tokens = input_ids.shape[-1]
        max_new_tokens = generation_config.get('max_new_tokens', 50)
        
        # Token timing list
        token_times = []
        
        # Create streamer
        streamer = TextIteratorStreamer(
            model.config._name_or_path if hasattr(model.config, '_name_or_path') else 'tokenizer',
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # Update generation config
        gen_kwargs = {**generation_config, 'streamer': streamer}
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # Run generation in thread
        thread = Thread(target=model.generate, kwargs={'input_ids': input_ids, **gen_kwargs})
        thread.start()
        
        # Collect token timings
        first_token_time = None
        for _ in streamer:
            current_time = time.perf_counter()
            if first_token_time is None:
                first_token_time = current_time
            token_times.append(current_time)
        
        thread.join()
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        # Calculate metrics
        total_time_ms = (end - start) * 1000
        generated_tokens = len(token_times)
        
        if first_token_time is not None:
            ttft_ms = (first_token_time - start) * 1000
        else:
            ttft_ms = total_time_ms
            
        if generated_tokens > 1:
            # TPT is average time between tokens (excluding TTFT)
            inter_token_times = [token_times[i] - token_times[i-1] for i in range(1, len(token_times))]
            tpt_ms = sum(inter_token_times) / len(inter_token_times) * 1000
        elif generated_tokens == 1:
            tpt_ms = ttft_ms
        else:
            tpt_ms = 0.0
            
        tokens_per_second = generated_tokens / (total_time_ms / 1000) if total_time_ms > 0 else 0.0
        
        result = ThroughputResult(
            ttft_ms=round(ttft_ms, 3),
            tpt_ms=round(tpt_ms, 3),
            tokens_per_second=round(tokens_per_second, 2),
            total_tokens=prompt_tokens + generated_tokens,
            total_time_ms=round(total_time_ms, 3),
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens
        )
        
        self.throughput_measurements.append({
            "label": label,
            "timestamp_sec": round(time.time() - self.start_time, 3),
            **result.to_dict()
        })
        
        print(f"⏱️  [{label}] {tokens_per_second:.1f} tok/s | TTFT={ttft_ms:.1f}ms | TPT={tpt_ms:.1f}ms | {generated_tokens} tokens in {total_time_ms:.0f}ms")
        
        return result
    
    def __del__(self):
        """Cleanup: Shutdown NVML."""
        try:
            pynvml.nvmlShutdown()
        except:
            pass  # Ignore errors during cleanup


if __name__ == "__main__":
    # Example usage / smoke test
    print("Running VRAMProfiler smoke test...\n")
    print("=" * 60)
    
    import torch
    
    profiler = VRAMProfiler()
    
    # Test 1: Create small tensor
    x = torch.randn(1000, 1000, device='cuda')
    profiler.log_vram("1K x 1K Tensor (4 MB)")
    
    # Test 2: Create large tensor
    y = torch.randn(10000, 10000, device='cuda')
    profiler.log_vram("10K x 10K Tensor (400 MB)")
    
    # Test 3: Delete tensors
    del x, y
    torch.cuda.empty_cache()
    profiler.log_vram("After Cleanup")
    
    # =========================================================================
    # Throughput Test with GPT-2
    # =========================================================================
    print("\n" + "=" * 60)
    print("Testing Throughput Measurement with GPT-2...")
    print("=" * 60 + "\n")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load small model
        model_name = "gpt2"
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda"
        )
        model.eval()
        
        profiler.log_vram("GPT-2 Loaded")
        
        # Prepare input
        prompt = "The future of artificial intelligence"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        prompt_tokens = input_ids.shape[-1]
        max_new_tokens = 50
        
        print(f"\nPrompt: '{prompt}' ({prompt_tokens} tokens)")
        print(f"Generating: {max_new_tokens} tokens\n")
        
        # Test throughput measurement
        def generate():
            with torch.no_grad():
                return model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
        
        result = profiler.measure_generation_throughput(
            generate_fn=generate,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            label="GPT-2 Generation",
            warmup_runs=2
        )
        
        # Show generated text
        output_ids = generate()
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\nGenerated: {generated_text[:100]}...")
        
        # Cleanup
        del model, tokenizer
        torch.cuda.empty_cache()
        profiler.log_vram("After Model Cleanup")
        
        print("\n" + "-" * 60)
        print("Throughput Results:")
        print(f"  Tokens/s:      {result.tokens_per_second:.1f}")
        print(f"  TTFT (est.):   {result.ttft_ms:.1f} ms")
        print(f"  TPT:           {result.tpt_ms:.1f} ms")
        print(f"  Generated:     {result.generated_tokens} tokens")
        print(f"  Total time:    {result.total_time_ms:.0f} ms")
        print("-" * 60)
        
    except ImportError as e:
        print(f"⚠️  Skipping throughput test (missing dependencies): {e}")
    except Exception as e:
        print(f"❌ Throughput test failed: {e}")
    
    # Save results
    profiler.save_to_json("profiler_test.json")
    
    print("\n" + "=" * 60)
    print("✅ Smoke test completed!")
    print(f"Peak VRAM delta: {profiler.get_peak_vram_mb():.1f} MB")
    print("=" * 60)
