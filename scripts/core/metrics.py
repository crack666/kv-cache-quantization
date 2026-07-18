"""Latency and throughput metrics using CUDA events.

Provides prefill-latency and decode-throughput measurements that are
independent of the profiler or KV-cache code, making them composable
in the profiler_suite orchestrator.
"""

import copy
import statistics
import time
from typing import Dict

import torch


class PrefillOverflowError(RuntimeError):
    """Raised by the prefill guard when the very first warmup forward
    already exceeds physical VRAM or a time limit (PCIe-swap symptom).

    Unter WDDM (Windows/WSL) gibt es keinen harten CUDA-OOM, solange das
    Commit-Limit nicht erreicht ist — stattdessen swappt der Treiber still
    über PCIe und ein 60-s-Forward wird zu Stunden.  Der Guard bricht so
    einen Kontext früh und explizit ab (Fix 2026-07-18).
    """

    def __init__(self, reason: str, peak_mb: float, elapsed_s: float):
        super().__init__(reason)
        self.peak_mb = peak_mb
        self.elapsed_s = elapsed_s


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
    measure_runs: int = 1,
    full_logits: bool = False,
    vram_budget_mb: float = 0.0,
    warmup_timeout_s: float = 0.0,
) -> Dict:
    """Measure prefill (prompt-processing) latency with CUDA events.

    By default the forward pass uses ``logits_to_keep=1`` so that, like
    ``model.generate()``, only the last position's logits are computed.
    Without this, a full forward materialises logits for every position
    ([seq, vocab] in BF16 plus an FP32 upcast), which inflates VRAM peaks
    by tens of GB for large-vocabulary models (Gemma: 262k tokens) and
    does not reflect deployment behaviour.  Protocol correction 2026-07-17;
    set ``full_logits=True`` to reproduce the legacy measurements.

    Args:
        model: HuggingFace CausalLM (already on device, eval mode).
        input_ids: ``[batch, seq_len]`` tensor on the model device.
        past_key_values: Optional pre-initialised cache object.
        warmup_runs: Number of untimed warm-up iterations.
        measure_runs: Number of timed runs; the reported latency is the
            median.  Extra (non-final) runs use cache copies so the
            original cache is only mutated once.  If the cache cannot
            be copied (e.g. quanto), extra runs are skipped and the
            effective run count is reported in ``n_runs``.
        vram_budget_mb: If > 0, raise :class:`PrefillOverflowError` when
            the VRAM peak after the first warmup exceeds this budget
            (physical VRAM → swap already active).
        warmup_timeout_s: If > 0, raise :class:`PrefillOverflowError`
            when the first warmup takes longer than this.

    Returns:
        dict with ``prefill_ms`` (median), ``prefill_ms_runs`` (all
        timed runs), ``n_runs``, ``tokens``, ``tokens_per_sec``, and
        the ``past_key_values`` produced by the final timed run.
    """

    # Generation-realistisches Prefill: nur Last-Token-Logits (wie generate()).
    # Fallback auf Voll-Forward, falls ein Modell den Parameter nicht kennt.
    fwd_kwargs = {} if full_logits else {"logits_to_keep": 1}

    def _forward(cache):
        try:
            return model(input_ids, past_key_values=cache, use_cache=True, **fwd_kwargs)
        except TypeError:
            fwd_kwargs.clear()
            return model(input_ids, past_key_values=cache, use_cache=True)

    # Warm-up (use copies so the original cache is not mutated)
    for i in range(warmup_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            _forward(_safe_cache_copy(past_key_values))
        torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - t0
        print(f"    prefill warmup {i + 1}/{warmup_runs}: "
              f"{elapsed_s * 1000:.0f} ms", flush=True)

        # Prefill-Guard: schon der erste Warmup zeigt, ob dieser Kontext
        # das VRAM-Budget sprengt oder im PCIe-Swap versinkt — dann sofort
        # abbrechen statt Stunden in unbrauchbare Messwerte zu versenken.
        if i == 0:
            peak_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)
                       if torch.cuda.is_available() else 0.0)
            if vram_budget_mb > 0 and peak_mb > vram_budget_mb:
                raise PrefillOverflowError(
                    f"VRAM-Peak {peak_mb:.0f} MB > physisch {vram_budget_mb:.0f} MB "
                    f"(Warmup {elapsed_s:.1f}s)", peak_mb, elapsed_s)
            if warmup_timeout_s > 0 and elapsed_s > warmup_timeout_s:
                raise PrefillOverflowError(
                    f"Warmup {elapsed_s:.0f}s > Limit {warmup_timeout_s:.0f}s "
                    f"(PCIe-Swap? Peak {peak_mb:.0f} MB)", peak_mb, elapsed_s)

    def _timed_forward(cache):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        with torch.no_grad():
            out = _forward(cache)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end), out

    times = []
    # Extra runs on cache copies (skipped when the cache is not copyable)
    for k in range(max(measure_runs - 1, 0)):
        cache_copy = _safe_cache_copy(past_key_values)
        if cache_copy is None and past_key_values is not None:
            continue  # cannot repeat without mutating the real cache
        elapsed, _ = _timed_forward(cache_copy)
        times.append(elapsed)
        print(f"    prefill run {k + 1}/{measure_runs}: {elapsed:.0f} ms", flush=True)

    # Final run on the real cache (its outputs are returned)
    elapsed, outputs = _timed_forward(past_key_values)
    times.append(elapsed)
    print(f"    prefill run {len(times)}/{measure_runs}: {elapsed:.0f} ms", flush=True)

    # Wirksamkeits-Check: logits_to_keep=1 muss die Logits auf die letzte
    # Position beschränken. Schluckt ein Modell den Parameter still (**kwargs
    # ohne Anwendung), würde der Voll-Forward unbemerkt zurückkehren.
    if not full_logits and fwd_kwargs and outputs.logits.shape[1] != 1:
        print(f"  WARNUNG: logits_to_keep=1 wirkungslos — logits.shape[1] = "
              f"{outputs.logits.shape[1]} (Voll-Forward-Protokoll aktiv!)")

    median_ms = statistics.median(times)
    n_tokens = input_ids.shape[-1]
    tokens_per_sec = (n_tokens / median_ms) * 1000 if median_ms > 0 else 0.0

    return {
        "prefill_ms": round(median_ms, 3),
        "prefill_ms_runs": [round(t, 3) for t in times],
        "n_runs": len(times),
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
    measure_runs: int = 1,
    step_timeout_s: float = 5.0,
    progress_every: int = 32,
) -> Dict:
    """Measure auto-regressive decode throughput with CUDA events.

    Generates ``n_tokens`` one-by-one (greedy) and returns aggregate
    timing.

    Watchdog (2026-07-17): Liegt die mittlere Schrittzeit nach den ersten
    8 Tokens über ``step_timeout_s`` (gesunde GPU: <0,3 s/Token; PCIe-Swap:
    >30 s/Token), wird der Lauf abgebrochen und ``aborted=True`` gemeldet,
    statt stundenlang unbeobachtet zu swappen. Fortschritt wird alle
    ``progress_every`` Tokens ausgegeben.

    Args:
        model: HuggingFace CausalLM (already on device, eval mode).
        input_ids: ``[batch, seq_len]`` prompt tensor on the model device.
        n_tokens: Number of tokens to decode.
        past_key_values: Optional pre-filled cache from a prefill step.
        warmup_runs: Number of untimed warm-up iterations (full decode loops).
        prefill_fn: Optional callable ``() -> cache`` that re-creates a
            filled cache.  Used when deepcopy of the cache is not possible
            (e.g. quanto).
        measure_runs: Number of timed decode loops; the reported timing
            is the median.

    Returns:
        dict with ``decode_ms`` (median), ``decode_ms_runs``, ``n_runs``,
        ``tokens``, ``tokens_per_sec``.
    """

    def _decode_loop(inp, cache, timed=False):
        """Greedy-Decode; gibt (cache, tokens_done, aborted) zurück."""
        cur = inp
        kv = cache
        loop_start = time.perf_counter()
        for i in range(n_tokens):
            with torch.no_grad():
                out = model(cur, past_key_values=kv, use_cache=True)
            kv = out.past_key_values
            cur = out.logits[:, -1:, :].argmax(dim=-1)
            done = i + 1
            if timed and done % progress_every == 0:
                el = time.perf_counter() - loop_start
                print(f"    decode {done}/{n_tokens} tokens "
                      f"({done / el:.2f} tok/s)", flush=True)
            # Watchdog: nach 8 Tokens hochrechnen, ob der Lauf gesund ist
            if done == 8:
                el = time.perf_counter() - loop_start
                if el / done > step_timeout_s:
                    print(f"    WATCHDOG: {el / done:.1f} s/Token nach {done} Tokens "
                          f"(Schwelle {step_timeout_s} s) — vermutlich PCIe-Swap, "
                          f"breche Decode ab.", flush=True)
                    return kv, done, True
        return kv, n_tokens, False

    def _get_cache_for_run(source_cache):
        """Return a usable cache copy for an extra timed run."""
        c = _safe_cache_copy(source_cache)
        if c is not None and hasattr(c, 'key_cache') and len(getattr(c, 'key_cache', [])) > 0:
            return c
        # deepcopy failed or produced empty cache — re-prefill
        if prefill_fn is not None:
            return prefill_fn()
        return _safe_cache_copy(source_cache)

    # Kernel-Warmup OHNE Kopie des gefüllten Caches: Ein Deepcopy hielte
    # 2x KV gleichzeitig im Speicher und verfälscht den VRAM-Peak
    # (Protokoll-Korrektur 2026-07-17). Stattdessen kurze Decode-Schleife
    # ab leerem Cache; quantisierungsspezifische Kernels wärmen sich in den
    # ersten Schritten des ersten Messlaufs auf (amortisiert über n_tokens).
    warm_tokens = min(8, n_tokens)
    for _ in range(warmup_runs):
        cur = input_ids
        kv = None
        for _ in range(warm_tokens):
            with torch.no_grad():
                out = model(cur, past_key_values=kv, use_cache=True)
            kv = out.past_key_values
            cur = out.logits[:, -1:, :].argmax(dim=-1)
        torch.cuda.synchronize()

    # Timed runs. Run 1 läuft auf dem ORIGINAL-Cache (keine Kopie):
    # Direkt danach wird der VRAM-Peak gelesen — er entspricht damit dem
    # Deployment-Verhalten (Modell + genau ein Cache). Weitere Runs nutzen
    # Kopien des (um n_tokens gewachsenen) Caches; deren verdoppelter
    # Speicherbedarf liegt NACH dem Snapshot, belastet aber randvolle
    # Karten real — deshalb ist measure_runs=1 für Decode der Default.
    tok_s_runs = []
    times = []
    vram_peak_first_run_mb = 0.0
    aborted = False
    tokens_first_run = n_tokens
    mutated_cache = past_key_values
    for run_idx in range(max(measure_runs, 1)):
        if run_idx == 0:
            run_cache = past_key_values
        else:
            run_cache = _get_cache_for_run(mutated_cache)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start.record()
        result_cache, tokens_done, run_aborted = _decode_loop(
            input_ids, run_cache, timed=True)
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        times.append(elapsed)
        tok_s_runs.append((tokens_done / elapsed) * 1000 if elapsed > 0 else 0.0)
        if run_idx == 0:
            mutated_cache = result_cache
            tokens_first_run = tokens_done
            aborted = run_aborted
            if torch.cuda.is_available():
                vram_peak_first_run_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            if run_aborted:
                break  # Swap erkannt — weitere Runs wären Zeitverschwendung

    median_ms = statistics.median(times)
    tokens_per_sec = statistics.median(tok_s_runs)

    return {
        "decode_ms": round(median_ms, 3),
        "decode_ms_runs": [round(t, 3) for t in times],
        "n_runs": len(times),
        "tokens": tokens_first_run,
        "tokens_per_sec": round(tokens_per_sec, 2),
        "vram_peak_first_run_mb": round(vram_peak_first_run_mb, 1),
        "aborted": aborted,
    }
