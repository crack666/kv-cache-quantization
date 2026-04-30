"""Unified model loader with attention-backend and KV-cache quantization support.

Extracted from quantize_kvcache_hf.py::run_experiment() model-loading logic,
extended with ``attn_implementation`` parameter for WisSem (attention-backend
comparison) and ``kv_quant`` for MA (KV-cache quantization).

Usage:
    model, tokenizer, info = load_model(
        "mistralai/Mistral-7B-v0.3",
        attn_backend="sdpa",
        kv_quant="int8-hqq",
    )
"""

import sys
from typing import Dict, Optional, Tuple

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# Supported attention backends — extend as new integrations land
_ATTN_BACKENDS = {"sdpa", "eager", "flash_attention_2", "sage"}


def _resolve_sage():
    """Enable SageAttention via the official plug-and-play monkey-patch.

    Replaces ``torch.nn.functional.scaled_dot_product_attention`` with
    ``sageattention.sageattn`` so that HF models using ``attn_implementation="sdpa"``
    transparently run through SageAttention INT8 kernels.

    Returns the attn_implementation string (``"sdpa"``) and a cleanup callable
    that restores the original SDPA function.
    """
    try:
        from sageattention import sageattn
    except ImportError:
        raise ImportError(
            "SageAttention requested but 'sageattention' package is not installed. "
            "Install via: pip install git+https://github.com/thu-ml/SageAttention.git --no-build-isolation"
        )

    import torch.nn.functional as F

    _original_sdpa = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = sageattn
    print("SageAttention 2.2: F.scaled_dot_product_attention → sageattn (INT8 QK, FP16/FP8 PV)")

    def _cleanup():
        F.scaled_dot_product_attention = _original_sdpa
        print("SageAttention: restored original F.scaled_dot_product_attention")

    return "sdpa", _cleanup


def _parse_kv_quant(kv_quant: str) -> Dict:
    """Parse a kv-quant spec string like 'int8-hqq' or 'int2-hqq-kivi' into a config dict.

    Supported formats:
        - 'none'           → no quantization
        - 'int8-hqq'       → symmetric quant (both axes same: hqq=1, quanto=0)
        - 'int4-quanto'    → symmetric quant
        - 'int2-hqq-kivi'  → KIVI-style asymmetric (keys per-channel, values per-token)

    The 'kivi' suffix switches to asymmetric axes:
        - axis_key=0 (per-channel: each channel gets its own scale)
        - axis_value=1 (per-token: each token position gets its own scale)
    This matches KIVI (Liu et al., ICML 2024) which found that key caches have
    per-channel outliers while value caches have per-token outliers.
    """
    if kv_quant is None or kv_quant == "none":
        return {"enabled": False, "nbits": None, "backend": None, "axis_key": None, "axis_value": None, "asymmetric": False}

    parts = kv_quant.lower().split("-")

    # Check for KIVI suffix: e.g. 'int2-hqq-kivi'
    asymmetric = False
    if len(parts) == 3 and parts[2] == "kivi":
        asymmetric = True
        parts = parts[:2]

    if len(parts) != 2 or not parts[0].startswith("int"):
        raise ValueError(
            f"Invalid kv_quant spec '{kv_quant}'. "
            "Expected format: 'int8-hqq', 'int4-quanto', 'int2-hqq-kivi', etc."
        )
    nbits = int(parts[0].replace("int", ""))
    backend = parts[1]
    if backend not in ("hqq", "quanto"):
        raise ValueError(f"Unknown quantization backend '{backend}'. Supported: hqq, quanto")
    if nbits not in (2, 4, 8):
        raise ValueError(f"Unsupported bit-width {nbits}. Supported: 2, 4, 8")

    # Axis selection: symmetric (default) vs asymmetric (KIVI)
    if asymmetric:
        # KIVI: keys per-channel (axis=0), values per-token (axis=1)
        axis_key = 0
        axis_value = 1
    else:
        # Symmetric default: hqq→1, quanto→0 (per HF recommendations)
        axis_key = 0 if backend == "quanto" else 1
        axis_value = 0 if backend == "quanto" else 1

    return {
        "enabled": True,
        "nbits": nbits,
        "backend": backend,
        "axis_key": axis_key,
        "axis_value": axis_value,
        "asymmetric": asymmetric,
    }


def _get_text_config(model_config):
    """Return the text-specific config, handling multimodal models with nested configs."""
    if hasattr(model_config, "text_config"):
        return model_config.text_config
    return model_config


def _collect_model_config(model) -> Dict:
    """Extract architecture metadata for JSON output."""
    cfg = _get_text_config(model.config)
    num_q_heads = getattr(cfg, "num_attention_heads", None)
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_q_heads)
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None and hasattr(cfg, "hidden_size") and num_q_heads:
        head_dim = cfg.hidden_size // num_q_heads

    gqa_ratio = f"{num_q_heads // num_kv_heads}:1" if num_q_heads and num_kv_heads else None
    num_params = sum(p.numel() for p in model.parameters())

    return {
        "num_params_b": round(num_params / 1e9, 2),
        "num_layers": getattr(cfg, "num_hidden_layers", None),
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "gqa_ratio": gqa_ratio,
        "head_dim": head_dim,
        "max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
    }


def _collect_environment() -> Dict:
    """Snapshot of software/hardware versions for reproducibility."""
    env: Dict = {
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": None,
        "gpu_vram_gb": None,
    }
    if torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    return env


def load_model(
    model_id: str,
    attn_backend: str = "sdpa",
    kv_quant: Optional[str] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, Dict]:
    """Load a HuggingFace causal-LM with configurable attention backend
    and optional KV-cache quantization.

    Args:
        model_id: HF repo id or local path.
        attn_backend: One of ``sdpa``, ``eager``, ``flash_attention_2``,
            ``sage``.
        kv_quant: ``None`` / ``"none"`` for FP16, or ``"int8-hqq"``,
            ``"int4-quanto"``, etc.
        device: Target device (``"cuda"`` or ``"cpu"``).
        dtype: Model weight dtype.

    Returns:
        ``(model, tokenizer, info_dict)`` where *info_dict* contains
        ``model_config``, ``environment``, ``kv_quant``, and
        ``attn_backend``.
    """
    if attn_backend not in _ATTN_BACKENDS:
        raise ValueError(f"Unknown attn_backend '{attn_backend}'. Supported: {_ATTN_BACKENDS}")

    kv_cfg = _parse_kv_quant(kv_quant)

    # Resolve attention implementation string
    attn_impl = attn_backend
    cleanup_fn = None
    if attn_backend == "sage":
        attn_impl, cleanup_fn = _resolve_sage()

    # Detect model type — multimodal models need a different Auto class
    auto_config = AutoConfig.from_pretrained(model_id)
    is_multimodal = hasattr(auto_config, "text_config")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model — use appropriate class for multimodal vs causal-only
    model_kwargs = dict(
        dtype=dtype,
        device_map=device if device != "cpu" else None,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
    )
    if is_multimodal:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
        print(f"  Multimodal model detected ({auto_config.model_type}) — loaded via AutoModelForImageTextToText")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if device == "cpu":
        model = model.to(device)
    model.eval()

    # For multimodal models, provide the text sub-config for cache creation
    text_config = _get_text_config(model.config)

    info = {
        "model_config": _collect_model_config(model),
        "environment": _collect_environment(),
        "attn_backend": attn_backend,
        "kv_quant": kv_cfg,
        "text_config": text_config,
        "is_multimodal": is_multimodal,
        "_cleanup_fn": cleanup_fn,
    }

    print(
        f"Loaded {model_id} | attn={attn_backend} | "
        f"kv_quant={'none' if not kv_cfg['enabled'] else kv_quant} | "
        f"params={info['model_config']['num_params_b']}B"
    )
    return model, tokenizer, info
