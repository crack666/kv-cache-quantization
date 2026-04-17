"""Perplexity evaluation on standard datasets (WikiText-2, PG-19).

Replaces the toy-text PPL from quantize_kvcache_hf.py with a proper
sliding-window evaluation on WikiText-2 (or optionally PG-19).

Supports two modes:
  - **Reference PPL** (``use_cache=False``): Standard sliding-window PPL
    without any cache.  The result is identical for every quant method
    and serves as the intrinsic model quality baseline.
  - **Quantized PPL** (``cache_factory`` provided): Each window is
    evaluated *through* the quantized cache so that dequantization
    errors propagate into the loss.  This measures the actual quality
    impact of KV-cache quantization.

Reference implementation follows the HuggingFace perplexity guide:
https://huggingface.co/docs/transformers/perplexity
"""

import torch
from typing import Callable, Optional


def compute_perplexity(
    model,
    tokenizer,
    dataset: str = "wikitext2",
    max_tokens: int = 4096,
    stride: Optional[int] = None,
    device: str = "cuda",
    cache_factory: Optional[Callable] = None,
) -> float:
    """Sliding-window perplexity on a standard dataset.

    Args:
        model: HuggingFace CausalLM (eval mode, on device).
        tokenizer: Matching tokenizer.
        dataset: ``"wikitext2"`` or ``"pg19"``.
        max_tokens: Maximum context window for evaluation.
        stride: Sliding-window stride. Defaults to ``max_tokens // 2``.
        device: Torch device string.
        cache_factory: Optional callable ``() -> cache`` that creates a
            fresh quantized cache for each window.  When provided the
            forward pass uses ``use_cache=True`` and feeds the resulting
            cache back, so dequantization errors affect the loss.

    Returns:
        Perplexity (float).
    """
    if stride is None:
        stride = max_tokens // 2

    text = _load_dataset_text(dataset)
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.size(1)

    nlls = []
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_tokens, seq_len)
        target_len = end - prev_end  # only score new tokens
        window_ids = input_ids[:, begin:end]
        window_len = window_ids.shape[1]

        trg_ids = window_ids.clone()
        # Mask tokens already scored in the previous window
        if target_len < window_len:
            trg_ids[:, :-target_len] = -100

        with torch.no_grad():
            if cache_factory is not None:
                # Two-phase evaluation: QuantizedCache.update() returns
                # ORIGINAL tensors on its first call (lazy init).
                # Dequantization errors only appear from the 2nd call
                # onward.  We must therefore split each window into
                # prefix (fills cache) → suffix (reads dequantized KV).
                context_len = window_len - target_len  # overlap from prev window

                if context_len >= 128:
                    # Enough overlap context to meaningfully fill cache.
                    cache = cache_factory()
                    # Phase 1: fill cache with context (overlap portion)
                    model(window_ids[:, :context_len],
                          past_key_values=cache, use_cache=True)
                    # Phase 2: score new tokens through dequantized cache
                    outputs = model(
                        window_ids[:, context_len:],
                        labels=trg_ids[:, context_len:],
                        past_key_values=cache,
                        use_cache=True,
                    )
                    neg_log_likelihood = outputs.loss * target_len
                else:
                    # First window or tiny context — no overlap to fill
                    # the cache, so a two-phase split would score
                    # different tokens and introduce a bias.  Fall back
                    # to a cacheless single pass (identical to reference).
                    outputs = model(window_ids, labels=trg_ids,
                                    use_cache=False)
                    neg_log_likelihood = outputs.loss * target_len
            else:
                outputs = model(window_ids, labels=trg_ids, use_cache=False)
                neg_log_likelihood = outputs.loss * target_len

        nlls.append(neg_log_likelihood if isinstance(neg_log_likelihood, float) else neg_log_likelihood.item())
        prev_end = end
        if end == seq_len:
            break

    total_tokens = prev_end
    ppl = torch.exp(torch.tensor(sum(nlls) / total_tokens)).item()
    return round(ppl, 4)


# -- dataset helpers ------------------------------------------------------

_DATASET_CACHE: dict = {}


def _load_dataset_text(name: str) -> str:
    """Load and concatenate the test split of a standard LM dataset."""
    if name in _DATASET_CACHE:
        return _DATASET_CACHE[name]

    from datasets import load_dataset

    if name == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
    elif name == "pg19":
        ds = load_dataset("pg19", split="test")
        # PG-19 test split can be very large; take first 5 books
        text = "\n\n".join(ds["text"][:5])
    else:
        raise ValueError(f"Unknown dataset '{name}'. Supported: wikitext2, pg19")

    _DATASET_CACHE[name] = text
    return text
