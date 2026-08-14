#!/usr/bin/env python3
"""Prueft die abgeleiteten Zahlen der Masterarbeit gegen die Messdaten.

Die Rohwerte der Arbeit stammen direkt aus den Messreihen und werden beim
Erzeugen der Tabellen uebernommen. Daneben stehen im Text jedoch Groessen, die
erst aus ihnen berechnet wurden: Prozentwerte, Verhaeltnisse und Faktoren. Sie
entstehen von Hand und driften, sobald eine Messreihe neu erhoben wird.

Dieses Skript rechnet sie aus results/tables/phase_b_long_context.csv nach und
vergleicht sie mit den in der Arbeit genannten Werten. Die statistischen
Kennwerte deckt statistical_tests.py ab.

Usage:
    python scripts/verify_derived_numbers.py
Exit 0 = alle Werte stimmen, Exit 1 = Abweichung.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).parent.parent
PHASE_B = BASE / "results" / "tables" / "phase_b_long_context.csv"
CTX = 32768
MB_PER_GB = 1024.0

MODELS = ["gemma-4-E4B", "Mistral-7B", "Yi-1.5-9B", "Qwen2-7B", "Qwen3-8B"]

checks = []


def check(label, computed, expected, tol):
    ok = abs(computed - expected) <= tol
    checks.append((ok, label, computed, expected))
    return ok


def load():
    at_ctx, by_ctx = defaultdict(dict), defaultdict(dict)
    with PHASE_B.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if int(row["context_len"]) == CTX:
                at_ctx[row["model"]][row["kv_quant"]] = row
            if row["kv_quant"] == "FP16":
                by_ctx[row["model"]][int(row["context_len"])] = row
    return at_ctx, by_ctx


def main():
    d, by_ctx = load()

    # --- Tabelle "VRAM-Peak bei 32768 Tokens" --------------------------------
    peaks = {"gemma-4-E4B": (24.1, 22.6, 21.8, 21.3), "Mistral-7B": (26.3, 22.6, 20.6, 19.6),
             "Yi-1.5-9B": (25.5, 22.7, 21.2, 20.4), "Qwen2-7B": (22.1, 20.4, 19.5, 19.1),
             "Qwen3-8B": (27.5, 23.3, 21.1, 19.9)}
    for m, want in peaks.items():
        for cfg, w in zip(("FP16", "INT8", "INT4", "INT2"), want):
            got = float(d[m][cfg]["vram_peak_mb"]) / MB_PER_GB
            check(f"VRAM-Peak {m} {cfg}", got, w, 0.05)

    # relative Einsparung INT2 und Verhaeltnis Peak- zu Cache-Reduktion
    rel = {"gemma-4-E4B": -11.5, "Mistral-7B": -25.2, "Yi-1.5-9B": -19.9,
           "Qwen2-7B": -13.4, "Qwen3-8B": -27.6}
    ratio = {"gemma-4-E4B": 1.87, "Mistral-7B": 1.96, "Yi-1.5-9B": 2.00,
             "Qwen2-7B": 2.00, "Qwen3-8B": 2.00}
    for m in MODELS:
        p16, p2 = (float(d[m][c]["vram_peak_mb"]) for c in ("FP16", "INT2"))
        c16, c2 = (float(d[m][c]["kv_cache_mb"]) for c in ("FP16", "INT2"))
        check(f"Einsparung INT2 {m}", (p2 - p16) / p16 * 100, rel[m], 0.06)
        check(f"dPeak/dCache {m}", (p16 - p2) / (c16 - c2), ratio[m], 0.006)

    # --- Kompressionsraten und Cache-Anteil ----------------------------------
    for m in MODELS:
        c = {cfg: float(d[m][cfg]["kv_cache_mb"]) for cfg in ("FP16", "INT8", "INT4", "INT2")}
        check(f"INT8-Rate {m}", c["INT8"] / c["FP16"] * 100, 53, 0.5)
        check(f"INT4-Rate {m}", c["INT4"] / c["FP16"] * 100, 28, 0.5)
        check(f"INT2-Rate {m}", c["INT2"] / c["FP16"] * 100, 15.6, 0.1)
        check(f"Cache je Token {m} (KiB)", c["FP16"] * 1024 / CTX,
              {"gemma-4-E4B": 56, "Mistral-7B": 128, "Yi-1.5-9B": 96,
               "Qwen2-7B": 56, "Qwen3-8B": 144}[m], 0.5)

    share = {"gemma-4-E4B": 7, "Mistral-7B": 15, "Qwen3-8B": 16}
    for m, w in share.items():
        got = float(d[m]["FP16"]["kv_cache_mb"]) / float(d[m]["FP16"]["vram_peak_mb"]) * 100
        check(f"Cache-Anteil am Peak {m}", got, w, 0.5)

    # --- Durchsatz ------------------------------------------------------------
    pf = {cfg: [(float(d[m][cfg]["prefill_ms"]) / float(d[m]["FP16"]["prefill_ms"]) - 1) * 100
                for m in MODELS] for cfg in ("INT8", "INT4", "INT2")}
    for cfg, (lo, hi) in {"INT8": (10, 23), "INT4": (17, 31), "INT2": (19, 40)}.items():
        check(f"Prefill-Aufschlag {cfg} Minimum", min(pf[cfg]), lo, 1.0)
        check(f"Prefill-Aufschlag {cfg} Maximum", max(pf[cfg]), hi, 1.0)

    dec = {m: {cfg: (float(d[m][cfg]["decode_tok_s"]) / float(d[m]["FP16"]["decode_tok_s"]) - 1)
               * 100 for cfg in ("INT8", "INT4", "INT2")} for m in MODELS}
    strong = ["Yi-1.5-9B", "Qwen2-7B", "Qwen3-8B"]
    check("Decode INT8 stark, geringster Verlust", max(dec[m]["INT8"] for m in strong), -43, 1.0)
    check("Decode INT8 stark, groesster Verlust", min(dec[m]["INT8"] for m in strong), -52, 1.0)
    check("Decode INT2 groesster Verlust", min(dec[m]["INT2"] for m in MODELS), -62, 1.0)
    check("Decode INT8 Mistral", dec["Mistral-7B"]["INT8"], -18, 0.5)
    check("Decode INT4 Mistral", dec["Mistral-7B"]["INT4"], -24, 0.6)
    check("Decode INT4 Gemma", dec["gemma-4-E4B"]["INT4"], -5, 0.5)

    # --- KIVI gegen HQQ -------------------------------------------------------
    kivi = {"Qwen3-8B": 53.7, "Qwen2-7B": 15.8}
    for m, w in kivi.items():
        hqq = abs(float(d[m]["INT2"]["delta_ppl"]))
        asym = abs(float(d[m]["INT2-KIVI"]["delta_ppl"]))
        check(f"KIVI-Faktor {m}", hqq / asym, w, 0.15)
    for m, w in {"Mistral-7B": 2.7, "Yi-1.5-9B": 4.2}.items():
        hqq = abs(float(d[m]["INT2"]["delta_ppl"]))
        asym = abs(float(d[m]["INT2-KIVI"]["delta_ppl"]))
        check(f"KIVI schlechter {m}", asym / hqq, w, 0.1)

    # --- Prefill-Skalierung 4k -> 32k ----------------------------------------
    factors = [float(by_ctx[m][CTX]["prefill_ms"]) / float(by_ctx[m][4096]["prefill_ms"])
               for m in MODELS]
    check("Prefill-Skalierung Minimum", min(factors), 11.3, 0.1)
    check("Prefill-Skalierung Maximum", max(factors), 23.3, 0.1)

    # --- INT8 verlustfrei -----------------------------------------------------
    check("groesstes |dPPL| bei INT8",
          max(abs(float(d[m]["INT8"]["delta_ppl"])) for m in MODELS), 0.004, 0.001)

    failed = [c for c in checks if not c[0]]
    print(f"{len(checks)} abgeleitete Zahlen geprueft, {len(failed)} Abweichung(en)\n")
    for ok, label, got, want in checks:
        if not ok:
            print(f"  ABWEICHUNG  {label}: berechnet {got:.3f}, in der Arbeit {want}")
    if not failed:
        print("  Alle Werte stimmen mit den Messdaten ueberein.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
