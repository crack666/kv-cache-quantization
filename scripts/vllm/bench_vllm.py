#!/usr/bin/env python3
"""Gegenmessung: NVFP4 unter vLLM gegen GGUF unter Ollama.

Misst denselben Kennzahlensatz wie ``ollama/bench_ollama.py`` und nutzt dieselben
Qualitaetssonden, damit die Zahlen nebeneinander stehen koennen. Ausgabe im
gleichen Schema v2.0.

Zwei Unterschiede zur Ollama-Harness, die aus der API folgen:

- **Prefill und Decode werden ueber Streaming getrennt.** vLLM meldet keine
  Phasendauern wie Ollamas ``prompt_eval_duration``. Stattdessen wird die Zeit
  bis zum ersten Token als Prefill gewertet und der Rest als Decode. Das ist
  nicht exakt dasselbe -- in der TTFT steckt auch etwas Scheduling -- aber bei
  Prefills von einer bis vier Minuten faellt das nicht ins Gewicht.
- **Der Speicher kommt aus ``nvidia-smi``,** nicht aus Puffermeldungen. vLLM
  allokiert ueber ``--gpu-memory-utilization`` einen festen Anteil vorab; eine
  Aufschluesselung nach Gewichten und KV wie bei llama.cpp gibt es nicht. Der
  Wert ist damit gerade *nicht* mit den llama.cpp-Puffersummen vergleichbar,
  wohl aber mit den nvidia-smi-Gegenproben.

    python bench_vllm.py --contexts 32768 131072
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from typing import List, Optional

import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "ollama"))
import quality_probe  # noqa: E402  -- gleiche Sonden wie auf dem Ollama-Pfad

SCHEMA_VERSION = "2.0"
DEFAULT_DEPTHS = [0.1, 0.25, 0.5, 0.75, 0.9]

FILLER = (
    "Die Verwaltung des Gebaeudes plant die Wartung der Lueftungsanlage. "
    "Mitarbeiterinnen und Mitarbeiter werden gebeten, die Hinweise am "
    "schwarzen Brett zu beachten. Der Empfang bleibt waehrend der "
    "Umbauphase durchgehend besetzt. "
)
NEEDLE_TEMPLATE = "Die geheime Zugangsnummer fuer das Rechenzentrum Berlin lautet {code}."
NEEDLE_QUESTION = (
    "\n\nFrage: Wie lautet die geheime Zugangsnummer fuer das Rechenzentrum Berlin? "
    "Antworte ausschliesslich mit der Zahl."
)


def vram_used_mib() -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--id=0", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return 0.0


def complete(host: str, model: str, prompt: str, max_tokens: int,
             seed: int = 42, timeout: float = 3600.0) -> dict:
    """Ein Streaming-Durchlauf. Trennt Prefill (TTFT) von Decode.

    Nutzt bewusst ``/v1/chat/completions`` und nicht ``/v1/completions``: nur der
    Chat-Endpunkt wendet das Chat-Template an und laesst den ``--reasoning-parser``
    greifen. Ueber den rohen Completion-Endpunkt landet das Reasoning als
    ``<think>...`` mitten im Antworttext und frisst das Token-Budget, bevor die
    eigentliche Antwort kommt -- gemessen: Needle 0/5 bei intaktem Modell.

    ``enable_thinking: false`` schaltet das Reasoning ganz ab, analog zu
    ``think: false`` auf dem Ollama-Pfad. Fuer den Durchsatz ist das unerheblich
    (tok/s normiert sich ueber die Tokenzahl), fuer die Retrieval-Sonden
    entscheidend.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{host.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")

    t0 = time.perf_counter()
    ttft: Optional[float] = None
    text_parts: List[str] = []
    usage = {}

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices") or []:
                delta = ch.get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text_parts.append(piece)
    total = time.perf_counter() - t0

    return {
        "response": "".join(text_parts),
        "ttft_s": ttft,
        "total_s": total,
        "prompt_tokens": usage.get("prompt_tokens"),
        "eval_count": usage.get("completion_tokens"),
    }


def build_body(target_tokens: int, chars_per_token: float, prefix: str = "",
               reserve: int = 1024) -> str:
    """``reserve`` haelt Platz frei fuer Frage UND Antwort.

    vLLM lehnt Anfragen mit HTTP 400 ab, wenn Prompt plus ``max_tokens`` die
    ``max_model_len`` uebersteigen -- anders als Ollama, das still abschneidet.
    Bei 131072 sind so alle Zellen gescheitert.
    """
    body_chars = int(max(target_tokens - reserve, 1) * chars_per_token)
    reps = max(int(body_chars / len(FILLER)) + 1, 1)
    body = (FILLER * reps)[:body_chars]
    if prefix:
        body = prefix + body[len(prefix):] if len(body) > len(prefix) else prefix + body
    return body


def build_haystack(target_tokens: int, cpt: float, depth: float, code: str,
                   prefix: str = "", reserve: int = 1024) -> str:
    body = build_body(target_tokens, cpt, prefix, reserve)
    needle = NEEDLE_TEMPLATE.format(code=code)
    cut = int(len(body) * depth)
    space = body.find(" ", cut)
    cut = space if space != -1 else cut
    return body[:cut] + " " + needle + " " + body[cut:] + NEEDLE_QUESTION


def needle_code(context_len: int, depth: float) -> str:
    return f"{(context_len // 1024) * 100 + int(depth * 100):06d}"


def calibrate(host: str, model: str) -> float:
    probe = FILLER * 40
    r = complete(host, model, probe, max_tokens=1)
    n = r.get("prompt_tokens")
    if not n:
        raise RuntimeError("vLLM meldet keine prompt_tokens -- Kalibrierung unmoeglich.")
    return len(probe) / n


def measure_context(host: str, model: str, ctx: int, cpt: float,
                    depths: List[float], decode_tokens: int,
                    warmup: int, runs: int, seed: int) -> dict:
    print(f"  [ctx={ctx}] Warmup ...", flush=True)
    for w in range(warmup):
        complete(host, model, build_body(ctx, cpt, f"[Warmup {w}] "), max_tokens=8)

    vram_after_load = vram_used_mib()

    print(f"  [ctx={ctx}] {runs} Messlaeufe ...", flush=True)
    prefill_tps, decode_tps, ttfts = [], [], []
    prompt_tokens = 0
    for i in range(runs):
        # Eindeutiger Praefix: vLLM hat Prefix-Caching, das sonst den Prefill
        # verfaelscht -- dieselbe Falle wie bei Ollamas Prompt-Cache.
        r = complete(host, model, build_body(ctx, cpt, f"[Lauf {i}] "),
                     max_tokens=decode_tokens, seed=seed + i)
        if not r["ttft_s"] or not r["eval_count"]:
            continue
        prompt_tokens = r["prompt_tokens"] or prompt_tokens
        ttfts.append(r["ttft_s"] * 1000)
        prefill_tps.append((r["prompt_tokens"] or 0) / r["ttft_s"])
        decode_time = max(r["total_s"] - r["ttft_s"], 1e-6)
        decode_tps.append(r["eval_count"] / decode_time)

    print(f"  [ctx={ctx}] Needle ueber {len(depths)} Tiefen ...", flush=True)
    hits, per_depth = 0, {}
    for depth in depths:
        code = needle_code(ctx, depth)
        r = complete(host, model,
                     build_haystack(ctx, cpt, depth, code, f"[Needle {depth}] "),
                     max_tokens=32, seed=seed)
        ans = r.get("response") or ""
        hit = code in re.sub(r"[^0-9]", "", ans)
        hits += int(hit)
        per_depth[str(depth)] = {"code": code, "hit": hit, "answer": ans.strip()[:120]}

    print(f"  [ctx={ctx}] Qualitaetssonden ...", flush=True)

    def _gen(prompt: str, num_predict: int, seed: int, think: bool = False) -> dict:
        # think wird ignoriert: vLLM trennt Reasoning serverseitig ueber
        # --reasoning-parser und liefert es nicht im Textfeld.
        return complete(host, model, prompt, max_tokens=num_predict, seed=seed)

    quality = quality_probe.run_all(_gen, ctx, build_body(ctx, cpt, "[Sonde] "), seed)

    return {
        "context_len": ctx,
        "prompt_tokens_measured": prompt_tokens,
        "ttft_ms": round(statistics.median(ttfts), 1) if ttfts else None,
        "prefill_tokens_per_sec": round(statistics.median(prefill_tps), 2) if prefill_tps else None,
        "decode_tokens_per_sec": round(statistics.median(decode_tps), 2) if decode_tps else None,
        "decode_tps_runs": [round(v, 2) for v in decode_tps],
        "decode_tokens": decode_tokens,
        "vram_after_load_mib": round(vram_after_load, 1),
        "quality": quality,
        "needle": {
            "trials": len(depths), "successes": hits,
            "success_rate": round(hits / len(depths), 4) if depths else None,
            "per_depth": per_depth,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description="NVFP4/vLLM-Gegenmessung")
    p.add_argument("--host", default="http://127.0.0.1:11436")
    p.add_argument("--model", default="qwen3.8-nvfp4")
    p.add_argument("--contexts", type=int, nargs="+", required=True)
    p.add_argument("--needle-depths", type=float, nargs="+", default=DEFAULT_DEPTHS)
    p.add_argument("--decode-tokens", type=int, default=512)
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--measure-runs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="../../results/ollama_probe/raw_vllm")
    p.add_argument("--label", default="nvfp4",
                   help="Kennung des Checkpoints fuer Dateiname und Ergebnis")
    args = p.parse_args()

    if ":11434" in args.host or ":11435" in args.host:
        print("ABBRUCH: 11434/11435 sind die Ollama-Instanzen.", file=sys.stderr)
        return 2

    try:
        urllib.request.urlopen(f"{args.host.rstrip('/')}/v1/models", timeout=15)
    except Exception as exc:
        print(f"ABBRUCH: vLLM unter {args.host} nicht erreichbar ({exc}).", file=sys.stderr)
        return 2

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"vllm_{args.label}_{ts}"

    print("=" * 72)
    print(f"vLLM NVFP4 | Host {args.host} | Kontexte {args.contexts}")
    print("=" * 72)
    print("Kalibriere Zeichen-je-Token ...")
    cpt = calibrate(args.host, args.model)
    print(f"  {cpt:.3f} Zeichen/Token")

    measurements = []
    for ctx in args.contexts:
        try:
            m = measure_context(args.host, args.model, ctx, cpt, args.needle_depths,
                                args.decode_tokens, args.warmup_runs,
                                args.measure_runs, args.seed)
            measurements.append(m)
            q = m["quality"]
            print(f"  -> prefill {m['prefill_tokens_per_sec']} tok/s | "
                  f"decode {m['decode_tokens_per_sec']} tok/s | "
                  f"VRAM {m['vram_after_load_mib']:.0f} MiB | "
                  f"needle {m['needle']['successes']}/{m['needle']['trials']} | "
                  f"multi {(q.get('multi_needle') or {}).get('hits')}/5 | "
                  f"verbatim {(q.get('verbatim') or {}).get('similarity')}")
        except Exception as exc:
            print(f"  -> FEHLER bei ctx={ctx}: {exc}", file=sys.stderr)
            measurements.append({"context_len": ctx, "error": str(exc)[:400],
                                 "error_type": type(exc).__name__})

    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "runtime": {
            "engine": "vllm",
            "host": args.host,
            "checkpoint": args.label,
            "note": "VRAM aus nvidia-smi, nicht aus Puffermeldungen -- vLLM "
                    "allokiert ueber gpu-memory-utilization vorab und schluesselt "
                    "nicht auf. Nicht mit den llama.cpp-Puffersummen vergleichbar.",
        },
        "config": {
            "seed": args.seed, "warmup_runs": args.warmup_runs,
            "measure_runs": args.measure_runs, "decode_tokens": args.decode_tokens,
            "needle_depths": args.needle_depths, "chars_per_token": round(cpt, 4),
        },
        "measurements": measurements,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, f"{experiment_id}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(f"\nGeschrieben: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
