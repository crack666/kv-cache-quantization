#!/usr/bin/env python3
"""Verifiziert alle Messwerte in den Thesis-Tabellen gegen die finalen JSONs.

Vergleicht Zelle für Zelle: Kap. 4 (tab:kv_vram, tab:vram_savings, tab:ppl_main,
tab:ppl_delta, tab:throughput, tab:kivi) und Kap. 5 (tab:overhead) gegen
results/raw/long_context_final/*_summary.json.

Konventionen (deklariert in Kap. 3, Hardware-Spezifikationen):
- Speicherwerte binär: JSON-"mb" = Bytes/2^20; Thesis-GB = mb/1024 (= GiB)
- Anzeige-Rundung kaufmännisch (round half up), nicht Pythons Banker's Rounding

Usage:
    python verify_thesis_numbers.py            # 0 Abweichungen = alles konsistent
"""

import glob
import json
import math
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "results/raw/long_context_final"
KV_DIST = BASE / "results/raw/kv_distributions_v2"
CH4 = BASE.parent / "thesis/chapters/04_results.tex"
CH5 = BASE.parent / "thesis/chapters/05_diskussion.tex"

MODEL_SHORT = {
    "gemma-4-E4B": "Gemma-4-E4B", "Mistral-7B-v0.1": "Mistral-7B",
    "Yi-1.5-9B": "Yi-1.5-9B", "Qwen2-7B": "Qwen2-7B", "Qwen3-8B": "Qwen3-8B",
}
Q = {"FP16": "fp16", "INT8": "int8-hqq", "INT4": "int4-hqq",
     "INT2": "int2-hqq", "KIVI": "int2-hqq(kivi)"}

fails, checks = [], 0


def round_half_up(x, digits):
    return float(Decimal(repr(x)).quantize(Decimal(f"1e-{digits}") if digits else Decimal("1"),
                                           rounding=ROUND_HALF_UP))


def chk(desc, tex_val, json_val, digits):
    """PASS, wenn der TEX-Wert eine korrekte Rundung des JSON-Werts ist:
    Abweichung maximal eine halbe Einheit der letzten angezeigten Stelle
    (toleriert half-up wie half-even an exakten ,5-Grenzen)."""
    global checks
    checks += 1
    tol = 0.5 * 10 ** -digits * 1.0001 if digits else 0.5
    if abs(tex_val - json_val) > tol:
        fails.append(f"  {desc}: TEX={tex_val} vs JSON={round_half_up(json_val, digits)} (roh {json_val})")


def table_rows(texstr, label):
    """Zeilen des Tabellen-Bodys zwischen \\label und \\end{table}."""
    i = texstr.index(label)
    body = texstr[i:texstr.index("\\end{table}", i)]
    out = {}
    for line in body.split("\n"):
        if "&" in line and "\\\\" in line and not line.strip().startswith(
                ("\\textbf", "%", "&", "\\multicolumn", "\\cmidrule")):
            cells = [c.strip().rstrip("\\").strip() for c in line.split("&")]
            out[re.sub(r"[\\~].*", "", cells[0]).strip()] = cells[1:]
    return out


def num(s):
    return float(re.sub(r"[^\d.\-+]", "",
                        s.replace("{,}", ".").replace(",", ".").replace("$", "")) or "nan")


def main():
    data = {}
    for f in sorted(glob.glob(str(DATA_DIR / "*_summary.json"))):
        d = json.load(open(f))
        short = MODEL_SHORT[d["model"].split("/")[-1]]
        data[short] = {c["kv_quant"]: c for c in d["combinations"]}
    if not data:
        sys.exit(f"Keine Summaries in {DATA_DIR}")

    tex4, tex5 = open(CH4).read(), open(CH5).read()

    for model, cells in table_rows(tex4, "label{tab:kv_vram}").items():
        if model not in data:
            continue
        chk(f"kv_vram {model} ctx", num(cells[0]), data[model]["fp16"]["ctx"], 0)
        for i, q in enumerate(["FP16", "INT8", "INT4", "INT2"]):
            chk(f"kv_vram {model} {q}", num(cells[i + 1]), data[model][Q[q]]["kv_mb"], 0)

    for model, cells in table_rows(tex4, "label{tab:vram_savings}").items():
        if model not in data:
            continue
        for i, q in enumerate(["FP16", "INT8", "INT4", "INT2"]):
            chk(f"vram {model} {q}", num(cells[i]), data[model][Q[q]]["vram_peak_mb"] / 1024, 1)
        dpct = (data[model][Q["INT2"]]["vram_peak_mb"] / data[model][Q["FP16"]]["vram_peak_mb"] - 1) * 100
        chk(f"vram {model} ΔINT2%", num(cells[4]), dpct, 1)

    for model, cells in table_rows(tex4, "label{tab:ppl_main}").items():
        if model not in data:
            continue
        chk(f"ppl {model} FP16", num(cells[0]), data[model]["fp16"]["ppl"], 3)
        for i, q in enumerate(["INT8", "INT4", "INT2", "KIVI"]):
            v = data[model][Q[q]]["ppl_quant"]
            chk(f"ppl {model} {q}", num(cells[i + 1]), v, 0 if v > 100 else (2 if v > 10 else 3))

    for model, cells in table_rows(tex4, "label{tab:ppl_delta}").items():
        if model not in data:
            continue
        ref = data[model]["fp16"]["ppl"]
        for i, q in enumerate(["INT8", "INT4", "INT2", "KIVI"]):
            # Vorzeichenbehaftet: negative Werte = Verbesserung gegenueber FP16
            d_json = data[model][Q[q]]["ppl_quant"] - ref
            if not math.isnan(num(cells[i])):
                mag = abs(d_json)
                chk(f"Δppl {model} {q}", num(cells[i]), d_json,
                    0 if mag > 100 else (1 if mag > 10 else 3))

    for model, cells in table_rows(tex4, "label{tab:throughput}").items():
        if model not in data:
            continue
        for i, q in enumerate(["FP16", "INT8", "INT4", "INT2"]):
            chk(f"prefill {model} {q}", num(cells[i]), data[model][Q[q]]["prefill_ms"], 0)
            chk(f"decode {model} {q}", num(cells[i + 4]), data[model][Q[q]]["decode_tok_s"], 1)

    # tab:kurtosis_summary gegen die Verteilungsdaten
    kvd = {}
    for f in sorted(glob.glob(str(KV_DIST / "*.json"))):
        d = json.load(open(f))
        kvd[MODEL_SHORT[d["model"].split("/")[-1]]] = d
    for model, cells in table_rows(tex4, "label{tab:kurtosis_summary}").items():
        if model not in kvd:
            continue
        s = kvd[model]["summary"]
        layers = kvd[model]["layers"]
        chk(f"kurtosis-mean {model}", num(cells[0]), s["key_kurtosis_mean"], 2)
        chk(f"kurtosis-max {model}", num(cells[1]), s["key_kurtosis_max"],
            1 if s["key_kurtosis_max"] > 10 else 2)
        # Anzeigegenauigkeit aus der Tabellenzelle ableiten (0.15 -> 2, 0.001 -> 3)
        m_dec = re.search(r"\.(\d+)", cells[2])
        chk(f"OR6sigma {model}", num(cells[2]), s["key_outlier_6sigma_mean"] * 100,
            len(m_dec.group(1)) if m_dec else 2)
        # "31/32" -> Heavy-Tail-Layer (Key-Kurtosis > 3) / Gesamtzahl
        heavy_tex, total_tex = (re.sub(r"[^\d/]", "", cells[3]).split("/") + ["nan"])[:2]
        heavy_json = sum(1 for l in layers if l["key"]["kurtosis"] > 3)
        chk(f"heavy-tail-layer {model}", float(heavy_tex), heavy_json, 0)
        chk(f"layer-gesamt {model}", float(total_tex), len(layers), 0)

    for model, cells in table_rows(tex4, "label{tab:kivi}").items():
        if model not in data:
            continue
        ref = data[model]["fp16"]["ppl"]
        for i, q in enumerate(["INT2", "KIVI"]):
            d_json = data[model][Q[q]]["ppl_quant"] - ref
            mag = abs(d_json)
            chk(f"kivi {model} {q}", num(cells[i]), d_json,
                0 if mag > 100 else (1 if mag > 10 else 3))

    m = data["Mistral-7B"]
    for q, cells in table_rows(tex5, "label{tab:overhead}").items():
        qq = {"FP16 (Baseline)": "fp16", "INT8 (HQQ)": "int8-hqq", "INT4 (HQQ)": "int4-hqq",
              "INT2 (HQQ)": "int2-hqq", "INT2 (KIVI)": "int2-hqq(kivi)"}.get(q.strip())
        if not qq:
            continue
        chk(f"overhead {q} prefill", num(cells[0]), m[qq]["prefill_ms"], 0)
        chk(f"overhead {q} decode", num(cells[1]), m[qq]["decode_tok_s"], 1)
        chk(f"overhead {q} vram", num(cells[2]), m[qq]["vram_peak_mb"] / 1024, 1)

    print(f"{checks} Zellen geprüft, {len(fails)} Abweichungen")
    for f in fails:
        print(f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
