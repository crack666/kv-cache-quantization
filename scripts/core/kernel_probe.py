"""SDPA effective-kernel verification.

``attn_implementation="sdpa"`` only selects PyTorch's dispatcher — which
kernel actually runs (Flash / cuDNN / Efficient / Math) is decided
silently at runtime based on the call signature (head_dim, mask type,
GQA flag, dtype).  Befund 2026-07-18 (Gemma-4-E4B): the 7 full-attention
layers call SDPA with head_dim 512 + ``enable_gqa=True`` — Flash (max
256) and cuDNN (max 128) are ineligible, Efficient rejects ``enable_gqa``
→ silent Math fallback materialising the N×N score matrix (21 GB per
layer at ctx 16k).  This probe turns the effective kernel from an
assumption into a measured, logged fact (``sdpa_effective_backend``).

Eligibility — not memory — is what is probed: kernel selection depends on
head_dim/dtype/mask/GQA-flag, not on sequence length, so a short probe
context is representative.
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

_FAST_BACKENDS = (
    ("flash", SDPBackend.FLASH_ATTENTION),
    ("cudnn", SDPBackend.CUDNN_ATTENTION),
    ("efficient", SDPBackend.EFFICIENT_ATTENTION),
)


def probe_sdpa_kernels(model, device: str = "cuda", probe_len: int = 1024) -> dict:
    """Spy on every ``F.scaled_dot_product_attention`` call during a short
    forward and test, per distinct call signature, which kernel the
    dispatcher can actually use (Math excluded).

    Returns a dict with ``signatures`` (one entry per distinct signature,
    incl. ``effective_backend``) and ``math_fallback`` (True if any
    signature can only run on the Math kernel).
    """
    orig = F.scaled_dot_product_attention
    seen = {}

    def spy(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
            scale=None, enable_gqa=False, **kw):
        sig = (
            q.shape[1], q.shape[-1], k.shape[1],
            "none" if attn_mask is None else str(attn_mask.dtype).replace("torch.", ""),
            bool(is_causal), bool(enable_gqa),
        )
        if sig not in seen:
            backend = "math"
            for name, be in _FAST_BACKENDS:
                try:
                    with sdpa_kernel(be):
                        orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                             is_causal=is_causal, scale=scale,
                             enable_gqa=enable_gqa, **kw)
                    backend = name
                    break
                except RuntimeError:
                    continue
            seen[sig] = {"count": 0, "backend": backend}
        seen[sig]["count"] += 1
        return orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                    is_causal=is_causal, scale=scale,
                    enable_gqa=enable_gqa, **kw)

    input_ids = torch.randint(1000, 5000, (1, probe_len), device=device)
    F.scaled_dot_product_attention = spy
    try:
        with torch.no_grad():
            try:
                model(input_ids, use_cache=False, logits_to_keep=1)
            except TypeError:
                model(input_ids, use_cache=False)
    finally:
        F.scaled_dot_product_attention = orig
    if device == "cuda":
        torch.cuda.empty_cache()

    signatures = []
    math_fallback = False
    for (heads, head_dim, kv_heads, mask, causal, gqa), info in seen.items():
        entry = {
            "q_heads": heads,
            "head_dim": head_dim,
            "kv_heads": kv_heads,
            "mask": mask,
            "is_causal": causal,
            "enable_gqa": gqa,
            "calls": info["count"],
            "effective_backend": info["backend"],
        }
        signatures.append(entry)
        if info["backend"] == "math":
            math_fallback = True

    print(f"  SDPA-Kernel-Probe ({len(signatures)} Signaturen, ctx {probe_len}):")
    for s in signatures:
        marker = "  ⚠ MATH-FALLBACK!" if s["effective_backend"] == "math" else ""
        print(f"    {s['calls']:3d}x head_dim={s['head_dim']} q/kv={s['q_heads']}/{s['kv_heads']} "
              f"mask={s['mask']} causal={s['is_causal']} gqa={s['enable_gqa']} "
              f"→ {s['effective_backend']}{marker}")
    if math_fallback:
        print("  ⚠ WARNUNG: Mindestens ein Layer-Typ läuft auf dem Math-Kernel — "
              "materialisierte N×N-Scores, Speicher-/Laufzeit-Explosion bei langen Kontexten.")

    return {
        "probe_len": probe_len,
        "signatures": signatures,
        "math_fallback": math_fallback,
    }
