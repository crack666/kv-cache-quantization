"""Needle-in-a-Haystack benchmark for KV-cache quantization.

Tests whether quantization destroys the model's ability to retrieve specific
information from long contexts. A known "needle" (fact) is inserted at various
depth positions within a long "haystack" (filler text), and the model is
prompted to complete a sentence that requires recalling the needle.

Design: Uses completion-style prompting ("The special magic number mentioned
in the text is:") rather than Q&A format. This is methodologically appropriate
for base (non-instruction-tuned) models which are trained to continue text,
not answer questions. Reference: Kamradt (2023), RULER (Hsieh et al., 2024).

Usage (standalone):
    python -m benchmarks.needle_haystack \
        --model mistralai/Mistral-7B-v0.1 \
        --kv-quant int4-hqq \
        --context-lengths 4096 8192 16384 \
        --depths 0.1 0.25 0.5 0.75 0.9

Integration with profiler_suite:
    from benchmarks.needle_haystack import run_needle_test
    results = run_needle_test(model, tokenizer, context_lengths=[4096, 8192],
                              kv_quant_cfg=kv_cfg, text_config=text_config)
"""

import re
import torch
from typing import Callable, Dict, List, Optional


# ── Default needle and retrieval prompt ───────────────────────────────────
NEEDLE = (
    "The special magic number for this experiment is 7492. "
    "Remember this number: 7492."
)
# Answer-prefix prompt (RULER-style, base-model friendly)
# Echoes the needle's sentence start so the model only needs to complete the value.
# Reference: Hsieh et al. (2024), "RULER", COLM 2024 (arXiv:2404.06654)
RETRIEVAL_PROMPT = (
    "The special magic number for this experiment is"
)
EXPECTED_ANSWER = "7492"

# Filler text — RULER noise sentences (Hsieh et al., 2024)
# Deliberately number-free to avoid distractor confusion with the numeric needle.
# Reference: NVIDIA/RULER, type_haystack='noise'
_FILLER_SENTENCES = [
    "The grass is green.",
    "The sky is blue.",
    "The sun is yellow.",
    "Here we go.",
    "There and back again.",
]


def _build_haystack(tokenizer, target_tokens: int, needle: str, depth_percent: float) -> str:
    """Build a haystack of approximately target_tokens with the needle at depth_percent.

    Args:
        tokenizer: HF tokenizer for length estimation.
        target_tokens: Desired total length in tokens.
        needle: The fact to hide in the text.
        depth_percent: 0.0 = beginning, 1.0 = end.

    Returns:
        The full text with needle inserted.
    """
    # Estimate tokens per filler sentence
    sample = " ".join(_FILLER_SENTENCES)
    sample_tokens = len(tokenizer.encode(sample, add_special_tokens=False))
    tokens_per_sentence = sample_tokens / len(_FILLER_SENTENCES)

    # How many filler sentences do we need?
    needle_tokens = len(tokenizer.encode(needle, add_special_tokens=False))
    filler_tokens_needed = target_tokens - needle_tokens
    n_sentences = max(1, int(filler_tokens_needed / tokens_per_sentence))

    # Build filler by cycling through sentences
    filler_sentences = []
    for i in range(n_sentences):
        filler_sentences.append(_FILLER_SENTENCES[i % len(_FILLER_SENTENCES)])

    # Insert needle at depth
    insert_idx = max(0, min(int(depth_percent * len(filler_sentences)), len(filler_sentences) - 1))
    filler_sentences.insert(insert_idx, needle)

    return " ".join(filler_sentences)


def _check_answer(generated_text: str, expected: str = EXPECTED_ANSWER) -> bool:
    """Check if the generated text contains the expected answer (case-insensitive).

    Follows RULER (Hsieh et al., 2024) string_match_all metric.
    """
    return expected.lower() in generated_text.lower()


def run_needle_test(
    model,
    tokenizer,
    context_lengths: List[int] = None,
    depths: List[float] = None,
    kv_quant_cfg: Optional[Dict] = None,
    text_config=None,
    needle: str = NEEDLE,
    retrieval_prompt: str = RETRIEVAL_PROMPT,
    expected: str = EXPECTED_ANSWER,
    max_new_tokens: int = 32,
    device: str = "cuda",
) -> Dict:
    """Run the Needle-in-a-Haystack test across context lengths and depths.

    Args:
        model: HF CausalLM model.
        tokenizer: Matching tokenizer.
        context_lengths: List of context lengths to test.
        depths: List of depth percentages (0.0 to 1.0).
        kv_quant_cfg: KV-quant config dict from _parse_kv_quant (or None for FP16).
        text_config: Model text config (needed for QuantizedCache).
        needle: The fact to hide.
        retrieval_prompt: Completion prompt appended after the haystack.
        expected: Expected answer string.
        max_new_tokens: Max tokens to generate for the answer.
        device: Torch device.

    Returns:
        Dict with results per (context_length, depth) combination.
    """
    if context_lengths is None:
        context_lengths = [4096, 8192, 16384]
    if depths is None:
        depths = [0.1, 0.25, 0.5, 0.75, 0.9]

    results = {
        "needle": needle,
        "retrieval_prompt": retrieval_prompt,
        "expected": expected,
        "format": "completion",
        "trials": [],
        "summary": {},
    }

    total_trials = len(context_lengths) * len(depths)
    successes = 0
    trial_idx = 0

    for ctx_len in context_lengths:
        for depth in depths:
            trial_idx += 1
            print(f"  [{trial_idx}/{total_trials}] ctx={ctx_len}, depth={depth:.0%}", end=" ")

            # Reserve space for the retrieval prompt + separator so it never gets truncated
            prompt_tokens = len(tokenizer.encode(f"\n\n{retrieval_prompt}", add_special_tokens=False))
            haystack_budget = ctx_len - prompt_tokens - 10  # 10 token safety margin

            # Build prompt: haystack + completion prompt
            haystack = _build_haystack(tokenizer, haystack_budget, needle, depth)
            prompt = f"{haystack}\n\n{retrieval_prompt}"

            # Tokenize (truncate to ctx_len to stay within budget)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=ctx_len,
            ).to(device)
            input_len = inputs["input_ids"].shape[-1]

            # Build cache
            cache = None
            if kv_quant_cfg and kv_quant_cfg.get("enabled"):
                from transformers import QuantizedCache
                cache = QuantizedCache(
                    backend=kv_quant_cfg["backend"],
                    config=text_config,
                    nbits=kv_quant_cfg["nbits"],
                    axis_key=kv_quant_cfg["axis_key"],
                    axis_value=kv_quant_cfg["axis_value"],
                    residual_length=kv_quant_cfg.get("residual_length", 128),
                )

            # Generate
            with torch.no_grad():
                gen_kwargs = {
                    "do_sample": False,
                    "max_new_tokens": max_new_tokens,
                    "use_cache": True,
                }
                if cache is not None:
                    gen_kwargs["past_key_values"] = cache

                outputs = model.generate(**inputs, **gen_kwargs)

            # Extract only generated tokens
            generated_ids = outputs[0, input_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Check
            success = _check_answer(generated_text, expected)
            if success:
                successes += 1

            status = "✓" if success else "✗"
            print(f"→ {status} ('{generated_text[:60]}')")

            results["trials"].append({
                "context_length": ctx_len,
                "depth_percent": depth,
                "input_tokens": input_len,
                "generated_text": generated_text[:200],
                "success": success,
            })

            # Cleanup
            del cache, outputs
            if device == "cuda":
                torch.cuda.empty_cache()

    # Summary
    results["summary"] = {
        "total_trials": total_trials,
        "successes": successes,
        "success_rate": round(successes / total_trials, 4) if total_trials > 0 else 0,
    }

    # Per-context summary
    per_ctx = {}
    for ctx_len in context_lengths:
        ctx_trials = [t for t in results["trials"] if t["context_length"] == ctx_len]
        ctx_successes = sum(1 for t in ctx_trials if t["success"])
        per_ctx[ctx_len] = {
            "trials": len(ctx_trials),
            "successes": ctx_successes,
            "success_rate": round(ctx_successes / len(ctx_trials), 4) if ctx_trials else 0,
        }
    results["summary"]["per_context"] = per_ctx

    print(f"\n  Needle-in-a-Haystack: {successes}/{total_trials} "
          f"({results['summary']['success_rate']:.0%} success rate)")

    return results


# ── Standalone CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json
    import sys
    from pathlib import Path

    # Add parent dir to path for imports
    _SCRIPT_DIR = Path(__file__).resolve().parent.parent
    if str(_SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPT_DIR))

    from core.model_loader import load_model

    parser = argparse.ArgumentParser(description="Needle-in-a-Haystack benchmark")
    parser.add_argument("--model", required=True, help="HF model id")
    parser.add_argument("--kv-quant", default="none", help="KV-quant spec (e.g. int4-hqq, int2-hqq-kivi)")
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75, 0.9])
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"Loading model: {args.model} (kv_quant={args.kv_quant})")
    model, tokenizer, info = load_model(
        args.model,
        attn_backend="sdpa",
        kv_quant=args.kv_quant if args.kv_quant != "none" else None,
        device=args.device,
        dtype=torch.float16,
    )

    results = run_needle_test(
        model, tokenizer,
        context_lengths=args.context_lengths,
        depths=args.depths,
        kv_quant_cfg=info["kv_quant"],
        text_config=info["text_config"],
        device=args.device,
    )

    # Add metadata
    results["model"] = args.model
    results["kv_quant"] = info["kv_quant"]
    results["seed"] = args.seed

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved: {out_path}")
    else:
        print(json.dumps(results["summary"], indent=2))
