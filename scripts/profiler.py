"""VRAM Profiler with Baseline Subtraction.

Measures GPU memory usage via NVML (NVIDIA Management Library) with automatic
baseline subtraction to isolate experiment VRAM from Windows background processes.

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
from typing import List, Dict, Optional


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
            "measurements": self.measurements
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"💾 Saved to {filepath}")
        print(f"   Peak VRAM: {output['peak_vram_delta_mb']:.1f} MB")
    
    def __del__(self):
        """Cleanup: Shutdown NVML."""
        try:
            pynvml.nvmlShutdown()
        except:
            pass  # Ignore errors during cleanup


if __name__ == "__main__":
    # Example usage / smoke test
    print("Running VRAMProfiler smoke test...\n")
    
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
    
    # Save results
    profiler.save_to_json("profiler_test.json")
    
    print("\n✅ Smoke test completed!")
    print(f"Peak VRAM delta: {profiler.get_peak_vram_mb():.1f} MB")
