#!/usr/bin/env python3
"""Erzeugt ausschliesslich layer_kurtosis_profile neu.

Der volle Plot-Lauf wuerde alle Abbildungen ueberschreiben. Die uebrigen
stammen aus dem Juli-Messlauf und bleiben deshalb unangetastet.
"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_thesis_plots as g  # noqa: E402

raw = {}
for f in glob.glob(str(g.KV_DIST / "*.json")):
    d = json.load(open(f))
    raw[g.MODEL_LABELS.get(d["model"], d["model"])] = d

print("Modelle:", sorted(raw))
for label, d in sorted(raw.items()):
    layers = d["layers"]
    ks = [l["key"]["kurtosis"] for l in layers]
    print("  %-16s n=%-3d  mean=%6.2f  max=%7.2f  peak@L%d"
          % (label, len(layers), sum(ks) / len(ks), max(ks),
             ks.index(max(ks))))

g.plot_layer_kurtosis(raw)
print("fertig")
