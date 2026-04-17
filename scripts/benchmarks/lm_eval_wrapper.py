"""Thin wrapper around lm-eval-harness for standardised benchmarks.

Supports MMLU, HellaSwag, and other tasks available in lm-eval ≥ 0.4.8.
Designed to work with models already loaded via core.model_loader so that
attention-backend and KV-cache quantization settings are preserved.
"""

from typing import Dict, List, Optional


def run_lm_eval(
    model,
    tokenizer,
    tasks: Optional[List[str]] = None,
    num_fewshot: Optional[int] = None,
    batch_size: str = "auto",
    device: str = "cuda",
    kv_quant_config: Optional[Dict] = None,
) -> Dict:
    """Run lm-eval-harness tasks and return scores.

    Args:
        model: HuggingFace CausalLM (already on device, eval mode).
        tokenizer: Matching tokenizer.
        tasks: List of task names (e.g. ``["mmlu"]``). Defaults to ``["mmlu"]``.
        num_fewshot: Number of few-shot examples. ``None`` = task default.
        batch_size: Batch size hint (``"auto"`` lets lm-eval decide).
        device: Device string for lm-eval.
        kv_quant_config: If provided, sets ``GenerationConfig.cache_implementation``
            and ``cache_config`` so that ``model.generate()`` uses quantized KV-cache.

    Returns:
        Dict with per-task results (accuracy, stderr, subtasks).
    """
    if tasks is None:
        tasks = ["mmlu"]

    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        raise ImportError(
            "lm-eval-harness is required for benchmark evaluation. "
            "Install via: pip install lm-eval>=0.4.8"
        )

    # Wrap the pre-loaded model in lm-eval's HFLM interface
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=str(device),
    )

    # If KV-cache quantization is active, inject into GenerationConfig
    if kv_quant_config and kv_quant_config.get("enabled"):
        _inject_kv_quant_generation_config(model, kv_quant_config)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
    )

    return _extract_scores(results, tasks)


def _inject_kv_quant_generation_config(model, kv_quant_config: Dict):
    """Set model.generation_config so generate() uses quantized KV-cache."""
    from transformers import GenerationConfig

    cache_config = {
        "backend": kv_quant_config["backend"],
        "nbits": kv_quant_config["nbits"],
    }

    if model.generation_config is None:
        model.generation_config = GenerationConfig()

    model.generation_config.cache_implementation = "quantized"
    model.generation_config.cache_config = cache_config


def _extract_scores(results: Dict, tasks: List[str]) -> Dict:
    """Normalise lm-eval output into our JSON v2 benchmark format."""
    out: Dict = {}
    raw = results.get("results", {})

    for task in tasks:
        task_data = raw.get(task, {})
        if not task_data:
            # Try with suffix (lm-eval sometimes uses "mmlu" → "mmlu_*")
            matching = {k: v for k, v in raw.items() if k.startswith(task)}
            if matching:
                # Aggregate: average across subtasks
                accs = [v.get("acc,none", v.get("acc_norm,none", 0)) for v in matching.values()]
                stderrs = [v.get("acc_stderr,none", v.get("acc_norm_stderr,none", 0)) for v in matching.values()]
                out[task] = {
                    "accuracy": round(sum(accs) / len(accs), 4) if accs else 0.0,
                    "stderr": round(sum(stderrs) / len(stderrs), 4) if stderrs else 0.0,
                    "subtasks": {k: round(v.get("acc,none", v.get("acc_norm,none", 0)), 4) for k, v in matching.items()},
                }
                continue

        acc = task_data.get("acc,none", task_data.get("acc_norm,none", 0))
        stderr = task_data.get("acc_stderr,none", task_data.get("acc_norm_stderr,none", 0))
        out[task] = {
            "accuracy": round(acc, 4) if acc else 0.0,
            "stderr": round(stderr, 4) if stderr else 0.0,
            "subtasks": {},
        }

    return out
