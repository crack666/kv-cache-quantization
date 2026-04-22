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
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    """Parse a kv-quant spec string like 'int8-hqq' into a config dict."""
    if kv_quant is None or kv_quant == "none":
        return {"enabled": False, "nbits": None, "backend": None}

    parts = kv_quant.lower().split("-")
    if len(parts) != 2 or not parts[0].startswith("int"):
        raise ValueError(
            f"Invalid kv_quant spec '{kv_quant}'. Expected format: 'int8-hqq', 'int4-quanto', etc."
        )
    nbits = int(parts[0].replace("int", ""))
    backend = parts[1]
    if backend not in ("hqq", "quanto"):
        raise ValueError(f"Unknown quantization backend '{backend}'. Supported: hqq, quanto")
    if nbits not in (2, 4, 8):
        raise ValueError(f"Unsupported bit-width {nbits}. Supported: 2, 4, 8")
    return {"enabled": True, "nbits": nbits, "backend": backend}


def _collect_model_config(model) -> Dict:
    """Extract architecture metadata for JSON output."""
    cfg = model.config
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

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device if device != "cpu" else None,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()

    info = {
        "model_config": _collect_model_config(model),
        "environment": _collect_environment(),
        "attn_backend": attn_backend,
        "kv_quant": kv_cfg,
        "_cleanup_fn": cleanup_fn,
    }

    print(
        f"Loaded {model_id} | attn={attn_backend} | "
        f"kv_quant={'none' if not kv_cfg['enabled'] else kv_quant} | "
        f"params={info['model_config']['num_params_b']}B"
    )
    return model, tokenizer, info
