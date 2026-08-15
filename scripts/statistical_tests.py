#!/usr/bin/env python3
"""Rechnet die statistischen Kennwerte der Masterarbeit aus den Messdaten nach.

Betroffen sind zwei Verfahren:

1. Spearman-Rangkorrelation zwischen mittlerer Key-Kurtosis und dem Betrag der
   Perplexitaetsdifferenz (H3a). Ausgewertet je Bitbreite (n = 5) und gepoolt
   ueber INT4 und INT2 (n = 10).
2. Mann-Whitney-U-Test zwischen den Sliding-Window- und den globalen Layern von
   Gemma-4-E4B (Abschnitt "Warum Gemma-4-E4B aus dem Muster faellt").
3. Exakter Vorzeichentest ueber die Needle-Versuche (Abschnitt
   "Perplexitaet als Qualitaetsmass"): Faellt das Retrieval unter
   Quantisierung haeufiger aus als unter FP16, als sich mit Zufall
   erklaeren laesst?
4. Spearman-Rangkorrelation der Layer-Kurtosis zwischen den beiden
   Kalibrationsstichproben (Abschnitt "Stabilitaet gegenueber der
   Kalibrationsstichprobe"): Bleibt die Rangfolge der Layer erhalten, wenn die
   Kennzahl an einem anderen Textausschnitt erhoben wird?

Zum p-Wert der Rangkorrelation: ``scipy.stats.spearmanr`` bestimmt ihn ueber
eine t-Approximation, die eine hinreichend grosse Stichprobe voraussetzt. Bei
n = 5 ist sie zu optimistisch. Dieses Skript berichtet daher zusaetzlich den
exakten Permutationstest, der alle 5! = 120 Rangzuordnungen aufzaehlt. Bei
n = 5 betraegt der kleinstmoegliche zweiseitige p-Wert 2/120 = 0.0167; kein
Ergebnis dieser Stichprobengroesse kann darunter liegen.

Datenquellen (alle im Repositorium):
  results/raw/kv_distributions_v2/kv_dist_*.json   Key-Kurtosis je Layer/Modell
  results/tables/phase_b_long_context.csv          Delta-PPL je Modell/Konfiguration
  results/raw/rotation/rotation_*.json             zweite Kalibrationsstichprobe
  results/tables/phase_b_long_context.csv          Needle-Trefferquote je Konfig.

Usage:
    python scripts/statistical_tests.py
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path(__file__).parent.parent
DIST_DIR = BASE / "results" / "raw" / "kv_distributions_v2"
PHASE_B = BASE / "results" / "tables" / "phase_b_long_context.csv"
ROT_DIR = BASE / "results" / "raw" / "rotation"

# Zweite Kalibrationsstichprobe. Von den drei Qwen2-Laeufen desselben Tages ist
# nur der letzte auswertbar; die beiden frueheren scheiterten an der
# Kontrollbedingung (siehe Fussnote im Ergebniskapitel).
ROTATION_RUNS = {
    "Qwen3-8B": "rotation_Qwen3-8B_int2_20260804_114123.json",
    "Qwen2-7B": "rotation_Qwen2-7B_int4_20260804_112454.json",
}

# Die Quellen benennen die Modelle unterschiedlich: Die Verteilungsmessungen
# fuehren den HuggingFace-Bezeichner, die Profiling-Tabelle einen Kurznamen.
# Beide werden auf die Schreibweise der Arbeit abgebildet.
MODELS = {
    "google/gemma-4-E4B": "Gemma-4-E4B",
    "mistralai/Mistral-7B-v0.1": "Mistral-7B",
    "01-ai/Yi-1.5-9B": "Yi-1.5-9B",
    "Qwen/Qwen2-7B": "Qwen2-7B",
    "Qwen/Qwen3-8B": "Qwen3-8B",
    "gemma-4-E4B": "Gemma-4-E4B",
    "Mistral-7B": "Mistral-7B",
    "Yi-1.5-9B": "Yi-1.5-9B",
    "Qwen2-7B": "Qwen2-7B",
    "Qwen3-8B": "Qwen3-8B",
}

# Needle-in-a-Haystack: vier Kontextlaengen mal fuenf Positionen je Konfiguration.
NEEDLE_TRIALS = 20

# Gemma-4-E4B: Layer mit eigenstaendigen KV-Tensoren. Die globalen Layer sind
# im Cache an der doppelten Key-Dimension erkennbar (512 statt 256).
GLOBAL_HEAD_DIM = 512


def load_kurtosis():
    """Mittlere Key-Kurtosis je Modell aus den Verteilungsmessungen."""
    out = {}
    for path in sorted(DIST_DIR.glob("kv_dist_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        ident = data["model"]
        if ident not in MODELS:
            raise KeyError(f"unbekannter Modellbezeichner in {path.name}: {ident}")
        out[MODELS[ident]] = data["summary"]["key_kurtosis_mean"]
    return out


def load_delta_ppl():
    """Betrag der Perplexitaetsdifferenz je Modell und Konfiguration.

    Die PPL ist kontextlaengen-unabhaengig; die Datei fuehrt denselben Wert je
    Kontextlaenge erneut auf. Wir nehmen den ersten belegten Eintrag und pruefen,
    dass die uebrigen damit uebereinstimmen.
    """
    out = {}
    with PHASE_B.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if not row["delta_ppl"]:
                continue
            model = MODELS.get(row["model"], row["model"])
            key = (model, row["kv_quant"])
            value = abs(float(row["delta_ppl"]))
            if key in out:
                assert abs(out[key] - value) < 1e-9, f"uneinheitliche Delta-PPL fuer {key}"
            out[key] = value
    return out


def load_needle():
    """Anzahl erfolgreicher Needle-Abrufe je Modell und Konfiguration.

    Die Tabelle fuehrt die ueber alle 20 Versuche aggregierte Trefferquote je
    Kontextlaenge erneut auf. Wir nehmen den ersten Eintrag und pruefen, dass
    die uebrigen damit uebereinstimmen.
    """
    out = {}
    with PHASE_B.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if not row.get("needle_score"):
                continue
            model = MODELS.get(row["model"], row["model"])
            key = (model, row["kv_quant"])
            hits = float(row["needle_score"]) * NEEDLE_TRIALS
            assert abs(hits - round(hits)) < 1e-9, f"keine ganze Trefferzahl fuer {key}"
            hits = int(round(hits))
            if key in out:
                assert out[key] == hits, f"uneinheitliche Needle-Quote fuer {key}"
            out[key] = hits
    return out


def sign_test_p(k):
    """Zweiseitiger exakter Vorzeichentest bei k einseitigen Diskordanzen.

    Alle k diskordanten Versuche zeigen in dieselbe Richtung; unter der
    Nullhypothese ist jede Richtung gleich wahrscheinlich. Der zweiseitige
    p-Wert ist damit 2 * 2^-k, fuer k = 0 nicht definiert.
    """
    if k == 0:
        return None
    return min(1.0, 2.0 * 2.0 ** -k)


def exact_spearman_p(x, y):
    """Zweiseitiger exakter Permutationstest fuer Spearmans rho.

    Zaehlt alle Zuordnungen der y-Werte zu den x-Werten auf und bestimmt den
    Anteil, dessen |rho| mindestens so gross ist wie der beobachtete. Ohne
    Naeherung und ohne Zufallsziehung.
    """
    def statistic(perm_y):
        return stats.spearmanr(x, perm_y).statistic

    result = stats.permutation_test(
        (y,), statistic,
        permutation_type="pairings",
        alternative="two-sided",
        n_resamples=np.inf,   # vollstaendige Aufzaehlung
    )
    return result.pvalue, result.null_distribution.size


def layer_kurtosis(model):
    """Key-Kurtosis je Layer aus der Hauptstichprobe (4096 Tokens)."""
    ident = next(k for k, v in MODELS.items() if v == model and "/" in k)
    for path in DIST_DIR.glob("kv_dist_*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["model"] == ident:
            return {l["layer"]: l["key"]["kurtosis"] for l in data["layers"]}
    raise FileNotFoundError(f"keine Verteilungsmessung fuer {model}")


def rotation_layer_kurtosis(model):
    """Key-Kurtosis je Layer aus der zweiten Stichprobe (Rotationsexperiment)."""
    data = json.loads((ROT_DIR / ROTATION_RUNS[model]).read_text(encoding="utf-8"))
    # Die Laeufe stammen aus zwei Schema-Staenden des Skripts: der aeltere nennt
    # die Felder "before"/"after", der neuere "kurt_before"/"kurt_after".
    per_layer = {}
    for entry in data["kurtosis_per_layer"]:
        value = entry.get("kurt_before", entry.get("before"))
        if value is None:
            raise KeyError(f"kein Kurtosis-Feld in {ROTATION_RUNS[model]}")
        per_layer[entry["layer"]] = value
    return per_layer, data["kurtosis_before_mean"]


def gemma_layer_groups():
    """Key-Kurtosis der Gemma-Layer, getrennt nach Attention-Typ."""
    path = next(DIST_DIR.glob("kv_dist_gemma*.json"))
    data = json.loads(path.read_text(encoding="utf-8"))
    sliding, glob = [], []
    for layer in data["layers"]:
        target = glob if layer["key_shape"][-1] == GLOBAL_HEAD_DIM else sliding
        target.append((layer["layer"], layer["key"]["kurtosis"]))
    return sliding, glob


def main():
    kurt = load_kurtosis()
    dppl = load_delta_ppl()
    models = ["Gemma-4-E4B", "Mistral-7B", "Yi-1.5-9B", "Qwen2-7B", "Qwen3-8B"]

    print("=" * 78)
    print("Eingangswerte")
    print("=" * 78)
    print(f"{'Modell':<14}{'kappa_quer':>12}{'|dPPL| INT4':>14}{'|dPPL| INT2':>14}")
    for m in models:
        print(f"{m:<14}{kurt[m]:>12.4f}{dppl[(m, 'INT4')]:>14.4f}{dppl[(m, 'INT2')]:>14.4f}")

    print()
    print("=" * 78)
    print("Spearman-Rangkorrelation: Key-Kurtosis gegen |Delta-PPL|")
    print("=" * 78)
    print(f"{'Auswertung':<20}{'n':>4}{'rho':>8}{'p (t-Approx.)':>16}{'p (exakt)':>12}{'Permut.':>10}")

    x_single = [kurt[m] for m in models]
    for label, cfg in [("INT4", "INT4"), ("INT2", "INT2")]:
        y = [dppl[(m, cfg)] for m in models]
        rho, p_t = stats.spearmanr(x_single, y)
        p_exact, n_perm = exact_spearman_p(x_single, y)
        print(f"{label:<20}{len(y):>4}{rho:>8.2f}{p_t:>16.4f}{p_exact:>12.4f}{n_perm:>10}")

    x_pool = x_single * 2
    y_pool = [dppl[(m, "INT4")] for m in models] + [dppl[(m, "INT2")] for m in models]
    rho, p_t = stats.spearmanr(x_pool, y_pool)
    print(f"{'gepoolt (INT4+INT2)':<20}{len(y_pool):>4}{rho:>8.2f}{p_t:>16.4f}"
          f"{'--':>12}{'zu gross':>10}")
    print()
    print("  Die gepoolte Auswertung enthaelt jeden Kurtosis-Wert doppelt. Diese")
    print("  Bindungen behandelt spearmanr ueber die Pearson-Korrelation der Raenge;")
    print("  ein exakter Permutationstest ist bei 10! Zuordnungen nicht sinnvoll.")
    print(f"  Kleinstmoeglicher zweiseitiger p-Wert bei n = 5: {2/120:.4f}")

    print()
    print("=" * 78)
    print("Mann-Whitney-U: Gemma-4-E4B, Sliding-Window- gegen globale Layer")
    print("=" * 78)
    sliding, glob = gemma_layer_groups()
    s_vals = [k for _, k in sliding]
    g_vals = [k for _, k in glob]
    print(f"  Sliding-Window ({len(s_vals):>2} Layer): Mittel {np.mean(s_vals):.2f}")
    print(f"  global         ({len(g_vals):>2} Layer): Mittel {np.mean(g_vals):.2f}"
          f"   Layer {[i for i, _ in glob]}")
    u, p = stats.mannwhitneyu(s_vals, g_vals, alternative="two-sided", method="exact")
    print(f"  U = {u:.1f}, p = {p:.4f} (exakt, zweiseitig)")
    print()
    print("  Die Richtung ist der Erwartung entgegengesetzt: Die globalen Layer")
    print("  haben die niedrigere Kurtosis, nicht die gefensterten.")

    print()
    print("=" * 78)
    print("Exakter Vorzeichentest: Needle-Abrufe FP16 gegen Quantisierung")
    print("=" * 78)
    needle = load_needle()
    print(f"{'Modell':<14}{'Konfig.':<12}{'FP16':>6}{'quant.':>8}{'k':>4}{'p':>10}")
    for m in models:
        base = needle.get((m, "FP16"))
        if base is None:
            continue
        for cfg in ("INT8", "INT4", "INT2", "INT2-KIVI"):
            hits = needle.get((m, cfg))
            if hits is None or hits == base:
                continue
            k = abs(base - hits)
            p = sign_test_p(k)
            print(f"{m:<14}{cfg:<12}{base:>6}{hits:>8}{k:>4}{p:>10.3f}")
    print()
    print("  k ist die Zahl der diskordanten Versuche. Sie folgt aus der Differenz\n"
          "  der Trefferzahlen, weil kein Versuch unter Quantisierung gelingt, den\n"
          "  FP16 verfehlt (Abschnitt 'Needle-in-a-Haystack'). Der Test setzt die\n"
          "  20 Versuche als unabhaengig an; da die Ausfaelle nach Kontextlaenge\n"
          "  klumpen, ist der p-Wert als Groessenordnung zu lesen.")

    print()
    print("=" * 78)
    print("Stabilitaet gegenueber der Kalibrationsstichprobe")
    print("=" * 78)
    print(f"{'Modell':<12}{'kappa 4096':>12}{'kappa 2048':>12}{'Abw.':>8}"
          f"{'rho (Layer)':>13}{'Layer':>7}{'argmax gleich':>15}")
    for model in ("Qwen3-8B", "Qwen2-7B"):
        main_layers = layer_kurtosis(model)
        rot_layers, rot_mean = rotation_layer_kurtosis(model)
        shared = sorted(set(main_layers) & set(rot_layers))
        a = [main_layers[i] for i in shared]
        b = [rot_layers[i] for i in shared]
        rho = stats.spearmanr(a, b).statistic
        main_mean = kurt[model]
        dev = (rot_mean - main_mean) / main_mean * 100
        same = (max(shared, key=lambda i: main_layers[i])
                == max(shared, key=lambda i: rot_layers[i]))
        print(f"{model:<12}{main_mean:>12.2f}{rot_mean:>12.2f}{dev:>+7.1f}%"
              f"{rho:>13.2f}{len(shared):>7}{'ja' if same else 'nein':>15}")
    print()
    print("  Die absoluten Werte haengen leicht von der Stichprobe ab, die")
    print("  Rangfolge der Layer dagegen kaum. Das traegt die Layer-Auswahl.")

    import scipy
    print()
    print(f"Software: scipy {scipy.__version__}, numpy {np.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
