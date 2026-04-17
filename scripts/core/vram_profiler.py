"""VRAM profiler with NVML baseline subtraction.

Consolidated from profiler.py (full VRAMProfiler with baseline subtraction).
The minimal version from quantize_kvcache_hf.py is superseded by this module.

Usage:
    profiler = VRAMProfiler()
    model = load_model().to("cuda")
    profiler.log_vram("Model loaded")
    outputs = model(inputs)
    profiler.log_vram("After forward")
    print(profiler.get_peak_vram_mb())
"""

import json
import time
from typing import Dict, List

import pynvml


class VRAMProfiler:
    """Measures GPU VRAM with baseline subtraction for scientific measurements.

    Baseline subtraction isolates experiment VRAM from OS / background
    processes that also occupy GPU memory (common on Windows with desktop
    compositing).
    """

    def __init__(self, gpu_index: int = 0):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.baseline_mb = self._measure_raw()
        self.measurements: List[Dict] = []
        self.start_time = time.time()

        device_name = pynvml.nvmlDeviceGetName(self.handle)
        if isinstance(device_name, bytes):
            device_name = device_name.decode()
        self.device_name = device_name
        print(f"VRAMProfiler initialized — GPU: {device_name}, Baseline: {self.baseline_mb:.1f} MB")

    # -- raw NVML access --------------------------------------------------

    def _measure_raw(self) -> float:
        """Raw VRAM usage in MB (includes OS / background)."""
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return info.used / 1e6

    # -- public API -------------------------------------------------------

    def measure_vram_mb(self) -> float:
        """VRAM delta in MB (experiment only, baseline subtracted)."""
        return self._measure_raw() - self.baseline_mb

    def get_vram_usage(self) -> Dict[str, float]:
        """Absolute VRAM usage as dict (used / free / total in GB)."""
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return {
            "used_gb": info.used / 1e9,
            "free_gb": info.free / 1e9,
            "total_gb": info.total / 1e9,
        }

    def log_vram(self, label: str):
        """Record a labelled measurement with timestamp."""
        delta = self.measure_vram_mb()
        raw = self._measure_raw()
        elapsed = time.time() - self.start_time

        measurement = {
            "label": label,
            "timestamp_sec": round(elapsed, 3),
            "vram_delta_mb": round(delta, 2),
            "vram_raw_mb": round(raw, 2),
            "vram_baseline_mb": round(self.baseline_mb, 2),
        }
        self.measurements.append(measurement)
        print(f"[{label}] Delta: {delta:.1f} MB | Raw: {raw:.1f} MB | t={elapsed:.1f}s")

    def get_peak_vram_mb(self) -> float:
        """Peak VRAM delta across all logged measurements."""
        if not self.measurements:
            return 0.0
        return max(m["vram_delta_mb"] for m in self.measurements)

    def save_to_json(self, filepath: str):
        """Persist measurements to a JSON file."""
        output = {
            "gpu": self.device_name,
            "baseline_mb": round(self.baseline_mb, 2),
            "peak_vram_delta_mb": round(self.get_peak_vram_mb(), 2),
            "measurements": self.measurements,
        }
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved VRAM log to {filepath}")

    def __del__(self):
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
