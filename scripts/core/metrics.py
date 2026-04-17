"""Latency and throughput metrics using CUDA events.

Provides prefill-latency and decode-throughput measurements that are
independent of the profiler or KV-cache code, making them composable
in the profiler_suite orchestrator.
"""

import copy
from typing import Dict

import torch


def _safe_cache_copy(cache):
    """Create a copy of a cache object, falling back gracefully.

    ``copy.deepcopy`` fails on quanto QuantizedCache objects because
    ``.clone()`` triggers JIT-compilation of CUDA extensions which
    require ``CUDA_HOME``.  When deepcopy fails we return *None* so
    the caller can use a ``prefill_fn`` to re-create the cache.
    """
    if cache is None:
        return None
    try:
        return copy.deepcopy(cache)
    except (OSError, ImportError, RuntimeError, AttributeError):
        return None


def measure_prefill_latency(
    model,
    input_ids: torch.Tensor,
    past_key_values=None,
    warmup_runs: int = 2,
) -> Dict:
    """Measure prefill (prompt-processing) latency with CUDA events.

    Args:
        model: HuggingFace CausalLM (already on device, eval mode).
        input_ids: ``[batch, seq_len]`` tensor on the model device.
        past_key_values: Optional pre-initialised cache object.
        warmup_runs: Number of untimed warm-up iterations.

    Returns:
        dict with ``prefill_ms``, ``tokens``, ``tokens_per_sec``, and
        the ``past_key_values`` produced by the timed run.
    """

    # Warm-up (use copies so the original cache is not mutated)
    for _ in range(warmup_runs):
        with torch.no_grad():
            model(input_ids, past_key_values=_safe_cache_copy(past_key_values), use_cache=True)
        torch.cuda.synchronize()

    # Timed run
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()
    with torch.no_grad():
        outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    n_tokens = input_ids.shape[-1]
    tokens_per_sec = (n_tokens / elapsed_ms) * 1000 if elapsed_ms > 0 else 0.0

    return {
        "prefill_ms": round(elapsed_ms, 3),
        "tokens": n_tokens,
        "tokens_per_sec": round(tokens_per_sec, 2),
        "past_key_values": outputs.past_key_values,
    }


def measure_decode_throughput(
    model,
    input_ids: torch.Tensor,
    n_tokens: int = 128,
    past_key_values=None,
    warmup_runs: int = 2,
    prefill_fn=None,
) -> Dict:
    """Measure auto-regressive decode throughput with CUDA events.

    Generates ``n_tokens`` one-by-one (greedy) and returns aggregate
    timing.

    Args:
        model: HuggingFace CausalLM (already on device, eval mode).
        input_ids: ``[batch, seq_len]`` prompt tensor on the model device.
        n_tokens: Number of tokens to decode.
        past_key_values: Optional pre-filled cache from a prefill step.
        warmup_runs: Number of untimed warm-up iterations (full decode loops).
        prefill_fn: Optional callable ``() -> cache`` that re-creates a
            filled cache.  Used when deepcopy of the cache is not possible
            (e.g. quanto).

    Returns:
        dict with ``decode_ms``, ``tokens``, ``tokens_per_sec``.
    """

    def _decode_loop(inp, cache):
        cur = inp
        kv = cache
        for _ in range(n_tokens):
            with torch.no_grad():
                out = model(cur, past_key_values=kv, use_cache=True)
            kv = out.past_key_values
            cur = out.logits[:, -1:, :].argmax(dim=-1)
        return kv

    def _get_cache_for_run():
        """Return a usable cache copy for a (warmup / timed) run."""
        c = _safe_cache_copy(past_key_values)
        if c is not None and hasattr(c, 'key_cache') and len(getattr(c, 'key_cache', [])) > 0:
            return c
        # deepcopy failed or produced empty cache — re-prefill
        if prefill_fn is not None:
            return prefill_fn()
        return _safe_cache_copy(past_key_values)

    # Warm-up
    for _ in range(warmup_runs):
        _decode_loop(input_ids, _get_cache_for_run())
        torch.cuda.synchronize()

    # Timed run
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()
    _decode_loop(input_ids, _get_cache_for_run())
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    tokens_per_sec = (n_tokens / elapsed_ms) * 1000 if elapsed_ms > 0 else 0.0

    return {
        "decode_ms": round(elapsed_ms, 3),
        "tokens": n_tokens,
        "tokens_per_sec": round(tokens_per_sec, 2),
    }
