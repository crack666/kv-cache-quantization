#!/usr/bin/env python3
"""Prueft, ob der protokollierte KV-Typ dem tatsaechlich gefahrenen entspricht.

Hintergrund: ``--kv-cache-type`` ist in ``bench_ollama.py`` nur eine
Beschriftung. Wirksam gesetzt wird der Typ ueber ``OLLAMA_KV_CACHE_TYPE`` im
Container. Laufen zwei Messlaeufe gleichzeitig gegen dieselbe Instanz, kann der
eine den Container unter dem anderen umkonfigurieren -- die Beschriftung stimmt
dann nicht mehr. Genau das ist am 2026-08-27 passiert, als ein per TaskStop
vermeintlich beendeter Lauf als Waisenprozess weiterlief.

Die von llama.cpp gemeldete KV-Puffergroesse ist dagegen Bodenwahrheit. Aus der
Architektur folgt sie eindeutig:

    16 KV-tragende Layer x 4 KV-Heads x 256 dim x (K+V) = 32768 Werte/Token

    f16   2.0000 Byte/Wert -> 64 KiB/Token
    q8_0  1.0625 Byte/Wert -> 34 KiB/Token   (32 Werte je 34-Byte-Block)
    q4_0  0.5625 Byte/Wert -> 18 KiB/Token   (32 Werte je 18-Byte-Block)

    python validate_runs.py            # nur pruefen
    python validate_runs.py --quarantine   # Fehlzuordnungen beiseiteschieben
"""

import argparse
import glob
import json
import os
import shutil
import sys

KIB_PER_TOKEN = {"f16": 64.0, "q8_0": 34.0, "q4_0": 18.0}
TOLERANCE = 0.02  # 2 % -- die Werte sind exakt, etwas Luft fuer Rundung


def expected_kv_mib(kv_type: str, context_len: int):
    kib = KIB_PER_TOKEN.get(kv_type)
    if kib is None:
        return None
    return kib * context_len / 1024.0


def actual_kv_type(kv_mib: float, context_len: int):
    """Rueckschluss vom gemessenen Puffer auf den tatsaechlichen Typ."""
    for name in KIB_PER_TOKEN:
        exp = expected_kv_mib(name, context_len)
        if exp and abs(kv_mib - exp) / exp <= TOLERANCE:
            return name
    return None


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="KV-Typ-Beschriftung gegen Messung pruefen")
    p.add_argument("--raw-dir", default=os.path.join(here, "..", "..",
                                                     "results", "ollama_probe", "raw"))
    p.add_argument("--quarantine", action="store_true",
                   help="Dateien mit Fehlzuordnung nach raw_mislabeled/ verschieben")
    args = p.parse_args()

    quarantine_dir = os.path.join(os.path.dirname(args.raw_dir), "raw_mislabeled")
    bad_files, ok_cells, bad_cells, unknown = [], 0, 0, 0

    for path in sorted(glob.glob(os.path.join(args.raw_dir, "*.json"))):
        try:
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception as exc:
            print(f"  nicht lesbar: {os.path.basename(path)} -- {exc}")
            continue

        declared = (d.get("kv_quant") or {}).get("kv_cache_type")
        file_bad = False
        for m in d.get("measurements", []):
            if "error" in m:
                continue
            ctx = m.get("context_len")
            kv_mib = (m.get("buffers") or {}).get("kv_mib")
            if not (ctx and kv_mib):
                unknown += 1
                continue
            real = actual_kv_type(kv_mib, ctx)
            if real is None:
                print(f"  ? {os.path.basename(path)} ctx={ctx}: "
                      f"KV {kv_mib} MiB passt zu keinem Typ")
                unknown += 1
            elif real != declared:
                print(f"  FEHLZUORDNUNG {os.path.basename(path)} ctx={ctx}: "
                      f"beschriftet '{declared}', gemessen '{real}' "
                      f"({kv_mib} MiB, erwartet {expected_kv_mib(declared, ctx):.0f})")
                bad_cells += 1
                file_bad = True
            else:
                ok_cells += 1
        if file_bad:
            bad_files.append(path)

    print(f"\n{ok_cells} Zellen stimmig, {bad_cells} fehlzugeordnet, {unknown} unpruefbar")

    if bad_files and args.quarantine:
        os.makedirs(quarantine_dir, exist_ok=True)
        for path in bad_files:
            shutil.move(path, os.path.join(quarantine_dir, os.path.basename(path)))
        print(f"{len(bad_files)} Dateien nach {quarantine_dir} verschoben")
    elif bad_files:
        print(f"{len(bad_files)} Dateien betroffen -- mit --quarantine verschieben")

    return 1 if bad_cells else 0


if __name__ == "__main__":
    sys.exit(main())
