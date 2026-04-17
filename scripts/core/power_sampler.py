"""Background GPU power sampling via NVML.

Extracted from profiler.py PowerSampler class. Provides a context-manager
interface for convenient power measurement during experiments.

Usage:
    with PowerSampler() as ps:
        # ... run experiment ...
        pass
    stats = ps.stats  # {"avg_watts": ..., "max_watts": ..., ...}
"""

import time
import threading
from typing import Dict, List, Optional

import pynvml


class PowerSampler:
    """Background thread for sampling GPU power with minimal overhead.

    Uses NVML ``nvmlDeviceGetPowerUsage()`` (~0.1 ms per call).
    Default sample interval: 50 ms (≈20 Hz).
    """

    def __init__(
        self,
        handle=None,
        gpu_index: int = 0,
        sample_interval_ms: int = 50,
    ):
        if handle is not None:
            self.handle = handle
            self._owns_nvml = False
        else:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self._owns_nvml = True

        self.sample_interval = sample_interval_ms / 1000.0
        self.samples: List[float] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.stats: Dict[str, float] = {}

    # -- context manager --------------------------------------------------

    def __enter__(self) -> "PowerSampler":
        self.start()
        return self

    def __exit__(self, *exc):
        self.stats = self.stop()

    # -- public API -------------------------------------------------------

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
            "samples": len(self.samples),
        }

    # -- internals --------------------------------------------------------

    def _sample_loop(self):
        while not self._stop_event.is_set():
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                self.samples.append(power_mw / 1000.0)  # mW → W
            except pynvml.NVMLError:
                pass
            time.sleep(self.sample_interval)

    def __del__(self):
        if self._owns_nvml:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
