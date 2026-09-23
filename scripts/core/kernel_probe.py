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

import warnings

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.profiler import ProfilerActivity, profile

_FAST_BACKENDS = (
    ("flash", SDPBackend.FLASH_ATTENTION),
    ("cudnn", SDPBackend.CUDNN_ATTENTION),
    ("efficient", SDPBackend.EFFICIENT_ATTENTION),
)

# Attention-relevante CUDA-Kernel-Namen → Backend-Klasse. Reihenfolge wichtig:
# cuDNN-Kernel können "fmha" enthalten, Efficient (cutlass) heißt "fmha_cutlassF...",
# Math zerfällt in gemm+softmax-Kernel (kein fused Attention-Kernel im Trace).
_KERNEL_CLASSES = (
    ("flash", ("flash",)),
    ("cudnn", ("cudnn",)),
    ("efficient", ("fmha", "efficient_attention")),
    ("math", ("softmax",)),
)


def _trace_winner(orig, q, k, v, **kw):
    """Run the exact SDPA call under torch.profiler (auto dispatch) and
    classify the winning kernel from the actual CUDA kernel names.

    Returns (backend_class, [kernel names]).  This is the definitive
    "which kernel ran" evidence — the eligibility test only proves
    which kernels *could* run.
    """
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        orig(q, k, v, **kw)
        torch.cuda.synchronize()  # Events flushen, sonst leerer Trace (Qwen-Bug 2026-07-18)
    names = [e.key for e in prof.key_averages() if e.device_type.name == "CUDA"]
    relevant = [n for n in names
                if any(t in n.lower() for _, tags in _KERNEL_CLASSES for t in tags)]
    joined = " ".join(names).lower()
    for backend, tags in _KERNEL_CLASSES:
        if any(t in joined for t in tags):
            return backend, relevant
    return "unknown", names


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
            call_kw = dict(attn_mask=attn_mask, dropout_p=dropout_p,
                           is_causal=is_causal, scale=scale, enable_gqa=enable_gqa, **kw)
            # 1) Gewinner: Trace der echten CUDA-Kernel (Auto-Dispatch).
            #    CUPTI ist bei wiederholten Profiler-Sessions im selben Prozess
            #    unzuverlässig (leerer Trace ab 2. Session) → einmal wiederholen.
            winner, kernel_names = _trace_winner(orig, q, k, v, **call_kw)
            if not kernel_names:
                winner, kernel_names = _trace_winner(orig, q, k, v, **call_kw)
            # 2) Wählbarkeit pro Fast-Kernel; PyTorch-Ablehnungsgründe als
            #    Warnungen einfangen (strukturiert ins JSON statt Log-Spam)
            eligible = []
            reasons = []
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                for name, be in _FAST_BACKENDS:
                    try:
                        with sdpa_kernel(be):
                            orig(q, k, v, **call_kw)
                        eligible.append(name)
                    except RuntimeError:
                        continue
            for w in caught:
                # "(Triggered internally at ...)" abschneiden, Boilerplate
                # ("kernel not used because:", "runtime disabled") verwerfen
                msg = str(w.message).split(" (Triggered internally")[0].strip()
                if (msg and "runtime disabled" not in msg
                        and not msg.endswith("not used because:")
                        and msg not in reasons):
                    reasons.append(msg)
            # 3) Leerer Trace ist kein Freifahrtschein: Ist kein Fast-Kernel
            #    wählbar, bleibt per Ausschluss zwingend der Math-Kernel.
            evidence = "trace"
            if winner == "unknown":
                if not eligible:
                    winner = "math"
                    evidence = "exclusion"
                else:
                    evidence = "eligibility-only"
            seen[sig] = {"count": 0, "backend": winner, "evidence": evidence,
                         "eligible": eligible, "kernels": kernel_names,
                         "rejection_reasons": reasons}
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
            "evidence": info["evidence"],
            "eligible_fast_kernels": info["eligible"],
            "cuda_kernels": info["kernels"],
            "rejection_reasons": info["rejection_reasons"],
        }
        signatures.append(entry)
        if info["backend"] == "math":
            math_fallback = True

    print(f"  SDPA-Kernel-Probe ({len(signatures)} Signaturen, ctx {probe_len}):")
    for s in signatures:
        if s["effective_backend"] == "math":
            verdict = ("math (per Ausschluss — kein Fast-Kernel wählbar)"
                       if s["evidence"] == "exclusion" else "math (Trace)")
            marker = "  ⚠ MATH-FALLBACK!"
        elif s["evidence"] == "eligibility-only":
            verdict = f"einer von: {', '.join(s['eligible_fast_kernels'])} (Trace leer)"
            marker = ""
        else:
            verdict = f"{s['effective_backend']} (wählbar: {', '.join(s['eligible_fast_kernels']) or 'keiner'})"
            marker = ""
        print(f"    {s['calls']:3d}x head_dim={s['head_dim']} q/kv={s['q_heads']}/{s['kv_heads']} "
              f"mask={s['mask']} causal={s['is_causal']} gqa={s['enable_gqa']} "
              f"→ {verdict}{marker}")
        for kn in s["cuda_kernels"]:
            print(f"        kernel: {kn}")
    if math_fallback:
        print("  ⚠ WARNUNG: Mindestens ein Layer-Typ läuft auf dem Math-Kernel — "
              "materialisierte N×N-Scores, Speicher-/Laufzeit-Explosion bei langen Kontexten.")

    return {
        "probe_len": probe_len,
        "signatures": signatures,
        "math_fallback": math_fallback,
    }
