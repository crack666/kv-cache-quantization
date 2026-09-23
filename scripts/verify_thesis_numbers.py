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


def header_cells(texstr, label):
    """Beschriftungen der Kopfzeile einer Tabelle, ohne die fuehrende Modellspalte.

    Die Kopfzeile ist die letzte Zeile vor \\midrule, die & und \\\\ enthaelt.
    LaTeX-Makros werden entfernt, sodass etwa \\textbf{OR}$_{3\\sigma}$ als
    "OR3sigma" zurueckkommt.
    """
    i = texstr.index(label)
    body = texstr[i:texstr.index("\\midrule", i)]
    head = [l for l in body.split("\n") if "&" in l and "\\\\" in l]
    if not head:
        return []
    cells = head[-1].split("&")[1:]
    out = []
    for c in cells:
        c = re.sub(r"\\[a-zA-Z]+", "", c)
        out.append(re.sub(r"[^A-Za-z0-9]", "", c))
    return out


def expect_columns(texstr, label, expected, name):
    """Meldet einen Befund, wenn sich die Spaltenstruktur einer Tabelle geaendert hat.

    Ohne diese Pruefung liest der Parser bei einer eingefuegten Spalte still die
    falsche Zelle -- er kann dadurch ebenso falsch bestehen wie falsch scheitern.
    """
    actual = header_cells(texstr, label)
    if actual != expected:
        fails.append("  STRUKTUR %s: Spalten geaendert\n"
                     "      erwartet: %s\n"
                     "      gefunden: %s\n"
                     "      -> Spaltenindizes in diesem Skript nachziehen"
                     % (name, expected, actual))
        return False
    return True


def check_raw_dirs_documented():
    """Meldet Rohdatenverzeichnisse, die results/raw/README.md nicht einordnet.

    Die Verzeichnisnamen geben die Reihenfolge der Messlaeufe nicht wieder
    ("v2" gegenueber "final"). Ohne diese Pruefung kann ein neuer Lauf
    unbemerkt neben dem Hauptdatensatz liegen.
    """
    raw = BASE / "results/raw"
    readme = raw / "README.md"
    if not readme.exists():
        fails.append("  README fehlt: %s" % readme)
        return
    text = readme.read_text(encoding="utf-8")
    for d in sorted(p.name for p in raw.iterdir() if p.is_dir()):
        if ("`%s/`" % d) not in text:
            fails.append("  Rohdatenverzeichnis nicht in results/raw/README.md "
                         "eingeordnet: %s" % d)


def main():
    check_raw_dirs_documented()
    data = {}
    for f in sorted(glob.glob(str(DATA_DIR / "*_summary.json"))):
        d = json.load(open(f))
        short = MODEL_SHORT[d["model"].split("/")[-1]]
        data[short] = {c["kv_quant"]: c for c in d["combinations"]}
    if not data:
        sys.exit(f"Keine Summaries in {DATA_DIR}")

    tex4, tex5 = open(CH4).read(), open(CH5).read()

    expect_columns(tex4, "label{tab:kv_vram}", ['Ctx', 'FP16', 'INT8', 'INT4', 'INT2', 'INT2Ratio'], "tab:kv_vram")
    for model, cells in table_rows(tex4, "label{tab:kv_vram}").items():
        if model not in data:
            continue
        chk(f"kv_vram {model} ctx", num(cells[0]), data[model]["fp16"]["ctx"], 0)
        for i, q in enumerate(["FP16", "INT8", "INT4", "INT2"]):
            chk(f"kv_vram {model} {q}", num(cells[i + 1]), data[model][Q[q]]["kv_mb"], 0)

    expect_columns(tex4, "label{tab:vram_savings}", ['FP16', 'INT8', 'INT4', 'INT2', 'INT2', 'PeakCache'], "tab:vram_savings")
    for model, cells in table_rows(tex4, "label{tab:vram_savings}").items():
        if model not in data:
            continue
        for i, q in enumerate(["FP16", "INT8", "INT4", "INT2"]):
            chk(f"vram {model} {q}", num(cells[i]), data[model][Q[q]]["vram_peak_mb"] / 1024, 1)
        dpct = (data[model][Q["INT2"]]["vram_peak_mb"] / data[model][Q["FP16"]]["vram_peak_mb"] - 1) * 100
        chk(f"vram {model} ΔINT2%", num(cells[4]), dpct, 1)

    expect_columns(tex4, "label{tab:ppl_main}", ['FP16', 'INT8', 'INT4', 'INT2', 'KIVI', ''], "tab:ppl_main")
    for model, cells in table_rows(tex4, "label{tab:ppl_main}").items():
        if model not in data:
            continue
        chk(f"ppl {model} FP16", num(cells[0]), data[model]["fp16"]["ppl"], 3)
        for i, q in enumerate(["INT8", "INT4", "INT2", "KIVI"]):
            v = data[model][Q[q]]["ppl_quant"]
            chk(f"ppl {model} {q}", num(cells[i + 1]), v, 0 if v > 100 else (2 if v > 10 else 3))

    expect_columns(tex4, "label{tab:ppl_delta}", ['INT8', 'INT4', 'INT2', 'KIVI', 'INT2', 'KIVI'], "tab:ppl_delta")
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

    QT = ["FP16", "INT8", "INT4", "INT2", "KIVI"]
    if expect_columns(tex4, "label{tab:throughput}", QT + QT, "tab:throughput"):
        for model, cells in table_rows(tex4, "label{tab:throughput}").items():
            if model not in data:
                continue
            for i, q in enumerate(QT):
                chk(f"prefill {model} {q}", num(cells[i]),
                    data[model][Q[q]]["prefill_ms"], 0)
                chk(f"decode {model} {q}", num(cells[i + len(QT)]),
                    data[model][Q[q]]["decode_tok_s"], 1)

    # tab:kurtosis_summary gegen die Verteilungsdaten
    kvd = {}
    for f in sorted(glob.glob(str(KV_DIST / "*.json"))):
        d = json.load(open(f))
        kvd[MODEL_SHORT[d["model"].split("/")[-1]]] = d
    # Spalten: kappa-mean, kappa-max, OR3sigma, OR6sigma, DR-mean, DR-max, VR, HT-Layer
    KS_COLS = ["Key", "", "OR3", "OR6", "DR", "DR", "VR", "HTL"]
    if expect_columns(tex4, "label{tab:kurtosis_summary}", KS_COLS, "tab:kurtosis_summary"):
      for model, cells in table_rows(tex4, "label{tab:kurtosis_summary}").items():
        if model not in kvd:
            continue
        s = kvd[model]["summary"]
        layers = kvd[model]["layers"]
        keys = [l["key"] for l in layers]

        def dec(cell, default=2):
            """Angezeigte Nachkommastellen der Zelle (0.15 -> 2, 0.001 -> 3)."""
            m = re.search(r"\.(\d+)", cell)
            return len(m.group(1)) if m else default

        chk(f"kurtosis-mean {model}", num(cells[0]), s["key_kurtosis_mean"], 2)
        chk(f"kurtosis-max {model}", num(cells[1]), s["key_kurtosis_max"],
            1 if s["key_kurtosis_max"] > 10 else 2)
        chk(f"OR3sigma {model}", num(cells[2]),
            sum(k["outlier_ratio_3sigma"] for k in keys) / len(keys) * 100, dec(cells[2]))
        chk(f"OR6sigma {model}", num(cells[3]), s["key_outlier_6sigma_mean"] * 100,
            dec(cells[3]))
        chk(f"DR-mean {model}", num(cells[4]),
            sum(k["dynamic_range"] for k in keys) / len(keys), dec(cells[4], 1))
        chk(f"DR-max {model}", num(cells[5]),
            max(k["dynamic_range"] for k in keys), dec(cells[5], 1))
        chk(f"VR {model}", num(cells[6]),
            sum(k["variance_ratio"] for k in keys) / len(keys), dec(cells[6]))
        # "31/32" -> Heavy-Tail-Layer (Key-Kurtosis > 3) / Gesamtzahl
        heavy_tex, total_tex = (re.sub(r"[^\d/]", "", cells[7]).split("/") + ["nan"])[:2]
        heavy_json = sum(1 for l in layers if l["key"]["kurtosis"] > 3)
        chk(f"heavy-tail-layer {model}", float(heavy_tex), heavy_json, 0)
        chk(f"layer-gesamt {model}", float(total_tex), len(layers), 0)

    expect_columns(tex4, "label{tab:kivi}", ['HQQ', 'KIVI', 'Faktor', 'Bewertung'], "tab:kivi")
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
    expect_columns(tex5, "label{tab:overhead}", ['Prefillms', 'Decodetoks', 'VRAMPeakGB'], "tab:overhead")
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
