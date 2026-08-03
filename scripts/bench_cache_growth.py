#!/usr/bin/env python3
"""Belegt den Faktor 2 im VRAM-Peak wachsender KV-Caches.

Hintergrund: HuggingFace Transformers laesst den KV-Cache durch Konkatenation
wachsen (``DynamicLayer.update``: ``self.keys = torch.cat([self.keys, key_states])``,
analog ``QuantizedLayer.update``). Dabei entsteht ein neuer Tensor voller Groesse,
waehrend der alte noch belegt ist -- der Cache liegt im Peak also doppelt vor.
``StaticLayer.update`` schreibt stattdessen per ``index_copy_`` in einen vorab
allozierten Puffer und vermeidet die Transiente.

Dieses Skript isoliert genau diesen Unterschied, ohne ein Modell zu laden, und
misst den tatsaechlichen CUDA-Peak beider Wachstumsmuster.

Erwartung: Variante A ~2.0x der Endgroesse, Variante B ~1.0x.

Usage:
    python bench_cache_growth.py
"""

import torch

B, H, D = 1, 32, 128          # batch, KV-Heads, head_dim
N = 8192                      # Zieltokens
DTYPE = torch.float16


def mib(num_bytes: float) -> float:
    return num_bytes / 2**20


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA nicht verfuegbar - der Test misst den GPU-Peak und wird uebersprungen.")
        return 1

    dev = "cuda"
    final_bytes = B * H * N * D * torch.finfo(DTYPE).bits // 8
    print(f"Zielgroesse des Caches: {mib(final_bytes):.1f} MiB "
          f"(B={B}, H={H}, N={N}, D={D}, {DTYPE})\n")

    # ── Variante A: torch.cat, wie DynamicLayer/QuantizedLayer ────────────
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    buf = torch.zeros(B, H, 0, D, device=dev, dtype=DTYPE)
    tok = torch.zeros(B, H, 1, D, device=dev, dtype=DTYPE)
    for _ in range(N):
        buf = torch.cat([buf, tok], dim=-2)
    torch.cuda.synchronize()

    peak_cat = torch.cuda.max_memory_allocated() - base
    del buf, tok
    torch.cuda.empty_cache()

    # ── Variante B: vorallozieren + in-place, wie StaticLayer ─────────────
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    buf = torch.zeros(B, H, N, D, device=dev, dtype=DTYPE)
    tok = torch.zeros(B, H, 1, D, device=dev, dtype=DTYPE)
    for i in range(N):
        buf[:, :, i:i + 1, :] = tok
    torch.cuda.synchronize()

    peak_pre = torch.cuda.max_memory_allocated() - base

    print(f"{'Variante':<36}{'Peak (MiB)':>12}{'Peak/Endgroesse':>18}")
    print(f"{'A: torch.cat (DynamicLayer)':<36}{mib(peak_cat):>12.1f}"
          f"{peak_cat / final_bytes:>18.2f}")
    print(f"{'B: prealloc + in-place (StaticLayer)':<36}{mib(peak_pre):>12.1f}"
          f"{peak_pre / final_bytes:>18.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
