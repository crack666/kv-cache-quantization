"""KV-cache utilities: size measurement, quantization timing, monkey-patching.

Consolidated from:
- profile_quant_overhead.py  — measure_kv_cache_size (better version with scale/zero)
- quantize_kvcache_hf.py     — QuantizationTimings, patch_quantized_cache
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch


# =============================================================================
# Quantization Timing Infrastructure
# =============================================================================

@dataclass
class QuantizationTimings:
    """Collects per-call timing data for quantize / dequantize operations."""

    quantize_times_ms: List[float] = field(default_factory=list)
    dequantize_times_ms: List[float] = field(default_factory=list)

    def add_quantize(self, ms: float):
        self.quantize_times_ms.append(ms)

    def add_dequantize(self, ms: float):
        self.dequantize_times_ms.append(ms)

    def summary(self) -> Dict:
        q_total = sum(self.quantize_times_ms)
        dq_total = sum(self.dequantize_times_ms)
        return {
            "quantize_total_ms": round(q_total, 3),
            "quantize_calls": len(self.quantize_times_ms),
            "quantize_avg_ms": round(q_total / len(self.quantize_times_ms), 3) if self.quantize_times_ms else 0,
            "dequantize_total_ms": round(dq_total, 3),
            "dequantize_calls": len(self.dequantize_times_ms),
            "dequantize_avg_ms": round(dq_total / len(self.dequantize_times_ms), 3) if self.dequantize_times_ms else 0,
            "total_overhead_ms": round(q_total + dq_total, 3),
        }

    def reset(self):
        self.quantize_times_ms.clear()
        self.dequantize_times_ms.clear()


# Module-level singleton ──────────────────────────────────────────────────────

_timings = QuantizationTimings()
_patches_applied = False


def get_timings() -> QuantizationTimings:
    return _timings


def reset_timings():
    _timings.reset()


@contextmanager
def measure_quantization_overhead():
    """Context manager that resets timings before and exposes them after the block."""
    reset_timings()
    yield _timings


# =============================================================================
# Monkey-patch for Quant/Dequant Timing
# =============================================================================

def patch_quantized_cache():
    """Monkey-patch HF QuantizedCache to instrument quantize/dequantize timing.

    Patches both ``QuantoQuantizedLayer`` and ``HQQQuantizedLayer``.
    Safe to call multiple times (idempotent).
    """
    global _patches_applied
    if _patches_applied:
        return

    try:
        from transformers.cache_utils import QuantoQuantizedLayer, HQQQuantizedLayer
    except ImportError:
        print("Warning: Could not import QuantizedLayer classes — overhead timing disabled")
        return

    # -- Quanto --
    _orig_quanto_q = QuantoQuantizedLayer._quantize
    _orig_quanto_dq = QuantoQuantizedLayer._dequantize

    def _timed_quanto_q(self, tensor, axis):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = _orig_quanto_q(self, tensor, axis)
        torch.cuda.synchronize()
        _timings.add_quantize((time.perf_counter() - t0) * 1000)
        return result

    def _timed_quanto_dq(self, qtensor):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = _orig_quanto_dq(self, qtensor)
        torch.cuda.synchronize()
        _timings.add_dequantize((time.perf_counter() - t0) * 1000)
        return result

    QuantoQuantizedLayer._quantize = _timed_quanto_q
    QuantoQuantizedLayer._dequantize = _timed_quanto_dq

    # -- HQQ --
    _orig_hqq_q = HQQQuantizedLayer._quantize
    _orig_hqq_dq = HQQQuantizedLayer._dequantize

    def _timed_hqq_q(self, tensor, axis):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = _orig_hqq_q(self, tensor, axis)
        torch.cuda.synchronize()
        _timings.add_quantize((time.perf_counter() - t0) * 1000)
        return result

    def _timed_hqq_dq(self, qtensor):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = _orig_hqq_dq(self, qtensor)
        torch.cuda.synchronize()
        _timings.add_dequantize((time.perf_counter() - t0) * 1000)
        return result

    HQQQuantizedLayer._quantize = _timed_hqq_q
    HQQQuantizedLayer._dequantize = _timed_hqq_dq

    _patches_applied = True
    print("QuantizedCache timing patches applied")


# =============================================================================
# KV-Cache Size Measurement
# =============================================================================

def measure_kv_cache_size(past_key_values) -> Tuple[float, str]:
    """Compute exact KV-cache size from PyTorch tensors.

    For ``QuantizedCache`` (HQQ / Quanto) this measures the *actual*
    compressed data including scale & zero-point metadata and the FP16
    residual buffer.

    Returns ``(size_mb, cache_type_str)``.
    """
    if past_key_values is None:
        return 0.0, "None"

    kv_cache_bytes = 0
    cache_type = type(past_key_values).__name__

    def _qbits_bytes(qbits):
        """Measure actual byte footprint of a quanto QBitsTensor.

        Quanto's QBitsTensor stores packed data in ``._data`` which is a
        ``PackedTensor``.  ``PackedTensor.numel()`` returns the *logical*
        element count, not the packed byte count — so we must drill into
        ``._data._data`` (the inner uint8 storage) for the true size.
        We also count ``._scale`` and ``._zeropoint`` metadata.
        """
        nbytes = 0
        packed = getattr(qbits, "_data", None)
        if packed is not None:
            inner = getattr(packed, "_data", None)
            if inner is not None:
                # Inner tensor is the actual packed uint8 storage
                nbytes += inner.numel() * inner.element_size()
            else:
                nbytes += packed.numel() * packed.element_size()
        for attr in ("_scale", "_zeropoint", "_shift"):
            meta = getattr(qbits, attr, None)
            if meta is not None and hasattr(meta, "numel"):
                nbytes += meta.numel() * meta.element_size()
        return nbytes

    # ── QuantizedCache with .layers ─────────────────────────────────────
    if hasattr(past_key_values, "layers") and len(past_key_values.layers) > 0:
        layer0 = past_key_values.layers[0]

        # HQQ / Quanto quantised layers
        if hasattr(layer0, "_quantized_keys"):
            for layer in past_key_values.layers:
                # Quantized keys: (qtensor, meta) tuple  ─ HQQ
                if hasattr(layer, "_quantized_keys") and layer._quantized_keys is not None:
                    qk = layer._quantized_keys
                    if isinstance(qk, tuple):
                        qtensor, meta = qk
                        kv_cache_bytes += qtensor.element_size() * qtensor.numel()
                        if isinstance(meta, dict):
                            for k in ("scale", "zero"):
                                if k in meta:
                                    kv_cache_bytes += meta[k].element_size() * meta[k].numel()
                    elif hasattr(qk, "_data"):  # Quanto QBits
                        kv_cache_bytes += _qbits_bytes(qk)
                    else:
                        kv_cache_bytes += qk.numel() * qk.element_size()

                # Quantized values
                if hasattr(layer, "_quantized_values") and layer._quantized_values is not None:
                    qv = layer._quantized_values
                    if isinstance(qv, tuple):
                        qtensor, meta = qv
                        kv_cache_bytes += qtensor.element_size() * qtensor.numel()
                        if isinstance(meta, dict):
                            for k in ("scale", "zero"):
                                if k in meta:
                                    kv_cache_bytes += meta[k].element_size() * meta[k].numel()
                    elif hasattr(qv, "_data"):  # Quanto QBits
                        kv_cache_bytes += _qbits_bytes(qv)
                    else:
                        kv_cache_bytes += qv.numel() * qv.element_size()

                # Residual FP16 buffer (recent tokens not yet quantized)
                if hasattr(layer, "keys") and layer.keys is not None and layer.keys.numel() > 0:
                    kv_cache_bytes += layer.keys.element_size() * layer.keys.numel()
                if hasattr(layer, "values") and layer.values is not None and layer.values.numel() > 0:
                    kv_cache_bytes += layer.values.element_size() * layer.values.numel()

            nbits = getattr(layer0, "nbits", "?")
            return kv_cache_bytes / (1024 * 1024), f"QuantizedCache (INT{nbits})"

        # DynamicCache with .layers (newer HF) — FP16
        elif hasattr(layer0, "keys") and hasattr(layer0, "values"):
            for layer in past_key_values.layers:
                if layer.keys is not None:
                    kv_cache_bytes += layer.keys.element_size() * layer.keys.numel()
                if layer.values is not None:
                    kv_cache_bytes += layer.values.element_size() * layer.values.numel()
            return kv_cache_bytes / (1024 * 1024), "DynamicCache (FP16)"

    # ── Legacy tuple format ──────────────────────────────────────────────
    try:
        for layer_kv in past_key_values:
            if isinstance(layer_kv, tuple) and len(layer_kv) >= 2:
                k, v = layer_kv[0], layer_kv[1]
                if isinstance(k, torch.Tensor):
                    kv_cache_bytes += k.element_size() * k.numel()
                if isinstance(v, torch.Tensor):
                    kv_cache_bytes += v.element_size() * v.numel()
        if kv_cache_bytes > 0:
            return kv_cache_bytes / (1024 * 1024), "DynamicCache (FP16, legacy)"
    except (TypeError, IndexError):
        pass

    return 0.0, f"{cache_type} (unknown structure)"
