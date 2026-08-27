#!/usr/bin/env python3
"""Benchmark-Harness fuer KV-Cache- und Gewichts-Quantisierung unter Ollama/llama.cpp.

Diese Harness gehoert NICHT zur Masterarbeit. Sie misst denselben Kennzahlensatz
wie ``profiler_suite.py`` (Prefill/Decode-Durchsatz, VRAM-Peak, Needle-Retrieval),
aber auf dem produktiv genutzten Stack: GGUF-Gewichte unter llama.cpp statt
FP16-Gewichten unter HuggingFace, und llama.cpp-KV-Cache-Typen (f16/q8_0/q4_0)
statt quanto/HQQ-``QuantizedCache``.

Der Grund fuer den eigenen Messpfad: Die Arbeit haelt die Gewichte bewusst in
FP16 und quantisiert ausschliesslich den Cache. Fuer ein 27B-Modell sind das
rund 54 GB allein an Gewichten, also mehr als die 32 GB der RTX 5090. Auf dieser
Karte ist die Methode der Arbeit auf dieses Modell nicht anwendbar; die
Gewichtsquantisierung ist hier keine Erweiterung, sondern Voraussetzung.

Die Ergebnisse sind deshalb NICHT direkt mit den Zahlen der Arbeit vergleichbar
-- anderer Quantisierer, andere Gewichtsprazision. Fuer die Bruecke dient der
Track-B-Lauf (``--model`` auf Mistral-7B bzw. Qwen3-8B in f16), bei dem die
Gewichtsprazision der Arbeit entspricht und nur die KV-Implementierung abweicht.

Ausgabeformat: JSON Schema v2.0, kompatibel zu ``results/raw/long_context_v2``,
mit einem zusaetzlichen ``runtime``-Block, der den llama.cpp-Pfad kennzeichnet.

Beispiel:
    python bench_ollama.py \
        --model qwen3.8:27b \
        --kv-cache-type q8_0 \
        --contexts 8192 32768 65536 131072 \
        --output-dir ../../results/ollama_probe/raw
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

import urllib.error
import urllib.request

import quality_probe

SCHEMA_VERSION = "2.0"
NS_PER_S = 1_000_000_000

# Needle-Protokoll wie in profiler_suite.py: fuenf Tiefen je Kontextlaenge.
DEFAULT_DEPTHS = [0.1, 0.25, 0.5, 0.75, 0.9]

NEEDLE_TEMPLATE = "Die geheime Zugangsnummer fuer das Rechenzentrum Berlin lautet {code}."
NEEDLE_QUESTION = (
    "\n\nFrage: Wie lautet die geheime Zugangsnummer fuer das Rechenzentrum Berlin? "
    "Antworte ausschliesslich mit der Zahl."
)

# Fuelltext fuer den Haystack. Bewusst generisch und ohne Zahlen, damit die
# Needle nicht zufaellig durch aehnliche Tokens maskiert wird.
FILLER = (
    "Die Verwaltung des Gebaeudes plant die Wartung der Lueftungsanlage. "
    "Mitarbeiterinnen und Mitarbeiter werden gebeten, die Hinweise am "
    "schwarzen Brett zu beachten. Der Empfang bleibt waehrend der "
    "Umbauphase durchgehend besetzt. "
)


# ---------------------------------------------------------------------------
# Ollama-API
# ---------------------------------------------------------------------------

def api_post(host: str, path: str, payload: dict, timeout: float = 1800.0) -> dict:
    """POST gegen die Ollama-API. Gibt das dekodierte JSON zurueck."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{host.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def api_get(host: str, path: str, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(f"{host.rstrip('/')}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def generate(
    host: str,
    model: str,
    prompt: str,
    num_ctx: int,
    num_predict: int,
    seed: int = 42,
    timeout: float = 1800.0,
    think: bool = True,
) -> dict:
    """Ein Generate-Durchlauf. Ollama liefert die Timings selbst mit.

    ``think`` steuert den Reasoning-Modus. qwen3.8 ist ein Thinking-Modell und
    gibt Reasoning in einem eigenen Feld ``thinking`` aus, das nicht in
    ``response`` landet. Fuer den Durchsatz bleibt Thinking an, weil das dem
    produktiven Betrieb entspricht und tok/s ueber ``eval_count`` ohnehin
    selbstnormalisierend ist. Fuer den Needle-Test wird es abgeschaltet, sonst
    misst man das Reasoning-Budget statt des Retrievals.
    """
    return api_post(
        host,
        "/api/generate",
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "30m",
            "think": think,
            "options": {
                "num_ctx": num_ctx,
                "num_predict": num_predict,
                "temperature": 0.0,
                "top_k": 1,
                "seed": seed,
            },
        },
        timeout=timeout,
    )


def unload(host: str, model: str) -> None:
    """Modell aus dem VRAM werfen, damit die naechste Zelle sauber misst."""
    try:
        api_post(host, "/api/generate", {"model": model, "keep_alive": 0}, timeout=120.0)
    except Exception:
        pass
    time.sleep(3.0)


# ---------------------------------------------------------------------------
# Puffer-Aufschluesselung aus dem llama.cpp-Log
# ---------------------------------------------------------------------------

# Geraetweites nvidia-smi ist eine naive Messung: Sie enthaelt den
# Desktop-Overhead, der hier je nach geoeffneten Fenstern um bis zu 3 GB
# schwankt. llama.cpp protokolliert seine Allokationen dagegen einzeln und
# exakt. Das ist auf diesem Pfad das Gegenstueck zur tensorbasierten Messung
# der Arbeit; nvidia-smi bleibt nur als Gegenprobe erhalten.
BUFFER_PATTERNS = [
    ("model_gpu_mib", r"load_tensors:\s+CUDA\d+ model buffer size\s*=\s*([\d.]+)"),
    ("model_cpu_mib", r"load_tensors:\s+CPU_Mapped model buffer size\s*=\s*([\d.]+)"),
    ("kv_mib", r"llama_kv_cache:\s+CUDA\d+ KV buffer size\s*=\s*([\d.]+)"),
    ("recurrent_state_mib", r"llama_memory_recurrent:\s+CUDA\d+ RS buffer size\s*=\s*([\d.]+)"),
    ("compute_gpu_mib", r"sched_reserve:\s+CUDA\d+ compute buffer size\s*=\s*([\d.]+)"),
    ("compute_host_mib", r"sched_reserve:\s+CUDA_Host compute buffer size\s*=\s*([\d.]+)"),
    ("clip_mib", r"load_hparams: model size:\s*([\d.]+)"),
]


def parse_llamacpp_buffers(container: str, since: str) -> dict:
    """Liest die Puffergroessen aus dem Container-Log seit ``since``.

    Mehrfachtreffer sind die Regel: qwen3.8 laedt neben dem Hauptkontext einen
    MTP-Kopf (``nextn_predict_layers``) und den CLIP-Projektor, die eigene
    Puffer melden. Der jeweils groesste Wert ist der Hauptkontext, die Summe
    aller Treffer die tatsaechliche Belegung -- beides wird ausgewiesen.
    """
    try:
        out = subprocess.run(
            ["docker", "logs", "--since", since, container],
            capture_output=True, text=True, timeout=60,
        )
    except Exception as exc:
        return {"error": str(exc)[:200]}

    log = (out.stdout or "") + (out.stderr or "")

    # Auf den LETZTEN Ladevorgang zuschneiden. Ein rein zeitbasiertes Fenster
    # reicht nicht: bei zu weitem Abstand geraten die Puffer frueherer
    # Kontexte mit in die Summe. Mit 3600s-Fenster gemessen: 36120 MiB statt
    # 18060, also glatt zwei Ladevorgaenge addiert. Jeder Load beginnt mit
    # llama_model_loader.
    marker = log.rfind("llama_model_loader:")
    if marker != -1:
        log = log[marker:]

    buffers = {}
    for name, pattern in BUFFER_PATTERNS:
        vals = [float(m) for m in re.findall(pattern, log)]
        if vals:
            buffers[name] = round(max(vals), 2)
            if len(vals) > 1:
                buffers[name + "_sum"] = round(sum(vals), 2)
                buffers[name + "_n"] = len(vals)

    gpu_keys = ("model_gpu_mib", "kv_mib", "recurrent_state_mib",
                "compute_gpu_mib", "clip_mib")
    total = sum(buffers.get(k + "_sum", buffers.get(k, 0.0)) for k in gpu_keys)
    if total:
        buffers["total_gpu_mib"] = round(total, 2)

    # Kontextunabhaengige Posten getrennt ausweisen: der SSM-State waechst
    # nicht mit der Kontextlaenge und gehoert nicht in die KV-Skalierung.
    if "kv_mib" in buffers:
        buffers["kv_context_scaling_mib"] = buffers["kv_mib"]

    if re.search(r"KV cache shifting is not supported", log):
        buffers["kv_cache_shifting"] = False
    return buffers


KIB_PER_TOKEN = {"f16": 64.0, "q8_0": 34.0, "q4_0": 18.0}


def verify_kv_type(buffers: dict, declared: str, context_len: int) -> dict:
    """Prueft, ob der tatsaechlich gefahrene KV-Typ der Beschriftung entspricht.

    ``--kv-cache-type`` ist nur ein Etikett; wirksam ist OLLAMA_KV_CACHE_TYPE im
    Container. Laeuft versehentlich ein zweiter Messlauf gegen dieselbe Instanz,
    kann er den Container umkonfigurieren und die Beschriftung wird still
    falsch. Am 2026-08-27 sind so fuenf Zellen entstanden, die q4_0 auswiesen,
    aber mit q8_0 liefen. Die von llama.cpp gemeldete Puffergroesse ist
    Bodenwahrheit, also wird hier dagegen geprueft.
    """
    kv_mib = buffers.get("kv_mib")
    if not kv_mib or not context_len:
        return {"kv_type_verified": None}

    measured = None
    for name, kib in KIB_PER_TOKEN.items():
        expected = kib * context_len / 1024.0
        if abs(kv_mib - expected) / expected <= 0.02:
            measured = name
            break

    if measured == declared:
        return {"kv_type_verified": True, "kv_type_measured": measured}

    print(f"    WARNUNG: KV-Typ stimmt nicht! beschriftet '{declared}', "
          f"gemessen '{measured}' ({kv_mib} MiB bei ctx={context_len}). "
          f"Laeuft ein zweiter Messlauf gegen dieselbe Instanz?", file=sys.stderr)
    return {"kv_type_verified": False, "kv_type_measured": measured}


def ollama_ps(host: str) -> dict:
    """Ollamas eigene Buchfuehrung ueber das geladene Modell."""
    try:
        data = api_get(host, "/api/ps", timeout=30)
    except Exception as exc:
        return {"error": str(exc)[:200]}
    models = data.get("models") or []
    if not models:
        return {}
    m = models[0]
    return {
        "size_bytes": m.get("size"),
        "size_vram_bytes": m.get("size_vram"),
        "size_vram_mib": round((m.get("size_vram") or 0) / 2**20, 1),
    }


# ---------------------------------------------------------------------------
# VRAM-Sampling
# ---------------------------------------------------------------------------

class VramSampler:
    """Pollt nvidia-smi im Hintergrund und haelt den Peak fest.

    Bewusst geraeteweit gemessen und nicht prozessweise: Der llama.cpp-Server
    laeuft im Container, und der Peak soll das Budget der Karte abbilden --
    genau die Groesse, an der die Pareto-Frage haengt.
    """

    def __init__(self, interval: float = 0.25, gpu_index: int = 0):
        self.interval = interval
        self.gpu_index = gpu_index
        self.peak_mb = 0.0
        self.baseline_mb = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def _read(gpu_index: int) -> float:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={gpu_index}",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return float(out.stdout.strip().splitlines()[0])
        except Exception:
            return 0.0

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, self._read(self.gpu_index))
            self._stop.wait(self.interval)

    def start(self) -> None:
        self.baseline_mb = self._read(self.gpu_index)
        self.peak_mb = self.baseline_mb
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return self.peak_mb


# ---------------------------------------------------------------------------
# Haystack-Aufbau
# ---------------------------------------------------------------------------

def calibrate_chars_per_token(host: str, model: str, num_ctx: int) -> float:
    """Ermittelt empirisch, wie viele Zeichen ein Token im Schnitt traegt.

    Ollama stellt keinen Tokenizer-Endpunkt bereit, meldet aber
    ``prompt_eval_count`` zurueck. Ein Probelauf genuegt daher, um den
    Haystack anschliessend auf die Ziel-Tokenzahl zu skalieren.
    """
    probe = FILLER * 40
    resp = generate(host, model, probe, num_ctx=num_ctx, num_predict=1)
    tokens = resp.get("prompt_eval_count", 0)
    if not tokens:
        raise RuntimeError(
            "Ollama meldet kein prompt_eval_count -- Kalibrierung nicht moeglich."
        )
    return len(probe) / tokens


def build_haystack(target_tokens: int, chars_per_token: float, depth: float, code: str,
                   prefix: str = "") -> str:
    """Baut einen Fuelltext der Zielgroesse mit der Needle an relativer Tiefe.

    ``prefix`` steht ganz vorne und dient dazu, Ollamas Prompt-Cache gezielt zu
    brechen. Ohne ihn trifft jeder Messlauf den Cache des Warmups: Ollama meldet
    dann zwar die volle ``prompt_eval_count``, aber nur die Dauer des
    ungecachten Rests -- der Prefill-Durchsatz wird dadurch um Groessenordnungen
    zu hoch ausgewiesen (bei 128k gemessen: 213920 statt realistischer Werte).
    """
    # Reserve fuer Frage und Needle, damit der Prompt das Fenster nicht sprengt.
    reserve_tokens = 96
    body_chars = int(max(target_tokens - reserve_tokens, 1) * chars_per_token)
    reps = max(int(body_chars / len(FILLER)) + 1, 1)
    body = (FILLER * reps)[:body_chars]

    if prefix:
        body = prefix + body[len(prefix):] if len(body) > len(prefix) else prefix + body

    needle = NEEDLE_TEMPLATE.format(code=code)
    cut = int(len(body) * depth)
    # Auf Satzgrenze ruecken, damit die Needle nicht mitten im Wort landet.
    space = body.find(" ", cut)
    cut = space if space != -1 else cut
    return body[:cut] + " " + needle + " " + body[cut:] + NEEDLE_QUESTION


def build_body(target_tokens: int, chars_per_token: float, prefix: str = "") -> str:
    """Reiner Fuelltext der Zielgroesse, ohne Needle und ohne Frage.

    Die Qualitaetssonden setzen ihre eigenen Inhalte hinein und brauchen
    deshalb einen unbeschriebenen Koerper.
    """
    reserve_tokens = 420  # Platz fuer Sondeninhalte und Frage
    body_chars = int(max(target_tokens - reserve_tokens, 1) * chars_per_token)
    reps = max(int(body_chars / len(FILLER)) + 1, 1)
    body = (FILLER * reps)[:body_chars]
    if prefix:
        body = prefix + body[len(prefix):] if len(body) > len(prefix) else prefix + body
    return body


def needle_code(context_len: int, depth: float) -> str:
    """Deterministischer, je Zelle eindeutiger Code."""
    return f"{(context_len // 1024) * 100 + int(depth * 100):06d}"


# ---------------------------------------------------------------------------
# Messung einer Zelle
# ---------------------------------------------------------------------------

def measure_context(
    host: str,
    model: str,
    context_len: int,
    chars_per_token: float,
    depths: List[float],
    decode_tokens: int,
    warmup_runs: int,
    measure_runs: int,
    seed: int,
    container: str,
    kv_cache_type: str,
) -> dict:
    """Misst Durchsatz, VRAM und Retrieval fuer eine Kontextlaenge."""

    # Sampler laeuft ab dem entladenen Zustand: die Baseline erfasst damit alles,
    # was sonst noch auf der Karte liegt, und der Peak schliesst den Ladevorgang
    # ein. Fuer die Budgetfrage zaehlt der Peak, nicht der eingeschwungene Wert.
    sampler = VramSampler()
    sampler.start()
    load_start = time.time()

    print(f"  [ctx={context_len}] Warmup ...", flush=True)
    for w in range(warmup_runs):
        wp = build_haystack(context_len, chars_per_token, 0.5,
                            needle_code(context_len, 0.5), prefix=f"[Warmup {w}] ")
        generate(host, model, wp, context_len, num_predict=8, seed=seed)

    # Direkt nach dem Laden auslesen, solange die Allokationszeilen dieses
    # Kontexts die juengsten im Log sind.
    # Fenster darf grosszuegig sein: die Praezision liefert der Zuschnitt auf
    # den letzten Ladevorgang, nicht die Zeitgrenze.
    since = f"{int(time.time() - load_start) + 120}s"
    buffers = parse_llamacpp_buffers(container, since)
    buffers.update(verify_kv_type(buffers, kv_cache_type, context_len))
    ps = ollama_ps(host)

    prefill_ms: List[float] = []
    decode_ms: List[float] = []
    prefill_tps: List[float] = []
    decode_tps: List[float] = []
    prompt_tokens_seen = 0

    print(f"  [ctx={context_len}] {measure_runs} Messlaeufe ...", flush=True)
    for i in range(measure_runs):
        # Eindeutiger Praefix je Lauf: sonst misst der Prefill den Prompt-Cache
        # statt der tatsaechlichen Verarbeitung.
        run_prompt = build_haystack(context_len, chars_per_token, 0.5,
                                    needle_code(context_len, 0.5),
                                    prefix=f"[Lauf {i}] ")
        resp = generate(
            host, model, run_prompt, context_len, num_predict=decode_tokens, seed=seed + i
        )
        pe_ns = resp.get("prompt_eval_duration", 0)
        pe_ct = resp.get("prompt_eval_count", 0)
        ev_ns = resp.get("eval_duration", 0)
        ev_ct = resp.get("eval_count", 0)
        prompt_tokens_seen = pe_ct or prompt_tokens_seen
        if pe_ns and pe_ct:
            prefill_ms.append(pe_ns / 1e6)
            prefill_tps.append(pe_ct / (pe_ns / NS_PER_S))
        if ev_ns and ev_ct:
            decode_ms.append(ev_ns / 1e6)
            decode_tps.append(ev_ct / (ev_ns / NS_PER_S))

    # Needle-Retrieval ueber alle Tiefen.
    print(f"  [ctx={context_len}] Needle ueber {len(depths)} Tiefen ...", flush=True)
    trials = 0
    successes = 0
    per_depth = {}
    for depth in depths:
        code = needle_code(context_len, depth)
        prompt = build_haystack(context_len, chars_per_token, depth, code,
                                prefix=f"[Needle {depth}] ")
        # think=False: sonst verbraucht das Reasoning das Token-Budget und
        # ``response`` bleibt leer, was faelschlich als Retrieval-Fehler zaehlt.
        resp = generate(
            host, model, prompt, context_len, num_predict=32, seed=seed, think=False
        )
        answer = resp.get("response", "") or ""
        hit = code in re.sub(r"[^0-9]", "", answer)
        trials += 1
        successes += int(hit)
        per_depth[str(depth)] = {"code": code, "hit": hit, "answer": answer.strip()[:120]}

    # Qualitaetssonden: feinkoerniger als der Einzel-Needle, siehe
    # quality_probe.py. Jede Sonde bekommt einen eigenen Praefix, damit sie
    # nicht den Prompt-Cache der vorigen trifft.
    print(f"  [ctx={context_len}] Qualitaetssonden ...", flush=True)

    def _gen(prompt: str, num_predict: int, seed: int, think: bool = False) -> dict:
        return generate(host, model, prompt, context_len, num_predict=num_predict,
                        seed=seed, think=think)

    probe_body = build_body(context_len, chars_per_token, prefix="[Sonde] ")
    quality = quality_probe.run_all(_gen, context_len, probe_body, seed)

    peak_mb = sampler.stop()

    return {
        "context_len": context_len,
        "prompt_tokens_measured": prompt_tokens_seen,
        "prefill_ms": round(statistics.median(prefill_ms), 3) if prefill_ms else None,
        "prefill_ms_runs": [round(v, 3) for v in prefill_ms],
        "prefill_n_runs": len(prefill_ms),
        "prefill_tokens_per_sec": round(statistics.median(prefill_tps), 2) if prefill_tps else None,
        "decode_ms": round(statistics.median(decode_ms), 3) if decode_ms else None,
        "decode_ms_runs": [round(v, 3) for v in decode_ms],
        "decode_n_runs": len(decode_ms),
        "decode_tokens": decode_tokens,
        "decode_tokens_per_sec": round(statistics.median(decode_tps), 2) if decode_tps else None,
        # Primaere Speichermessung: von llama.cpp selbst gemeldete Puffer.
        "buffers": buffers,
        "ollama_ps": ps,
        # Gegenprobe, geraetweit und inkl. Desktop-Overhead -- nicht primaer.
        "vram_peak_mb": round(peak_mb, 1),
        "vram_baseline_mb": round(sampler.baseline_mb, 1),
        "quality": quality,
        "needle": {
            "trials": trials,
            "successes": successes,
            "success_rate": round(successes / trials, 4) if trials else None,
            "per_depth": per_depth,
        },
    }


# ---------------------------------------------------------------------------
# Umgebung
# ---------------------------------------------------------------------------

def gpu_info() -> dict:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        name, total, driver = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
        return {"gpu_name": name, "vram_total_mb": float(total), "driver_version": driver}
    except Exception:
        return {"gpu_name": "unknown", "vram_total_mb": 0.0, "driver_version": "unknown"}


def model_info(host: str, model: str) -> dict:
    """Architekturangaben aus /api/show, soweit Ollama sie herausgibt."""
    try:
        show = api_post(host, "/api/show", {"model": model}, timeout=120.0)
    except Exception as exc:
        return {"error": str(exc)}

    details = show.get("details", {}) or {}
    info = show.get("model_info", {}) or {}

    def pick(*suffixes):
        for key, value in info.items():
            for suffix in suffixes:
                if key.endswith(suffix):
                    return value
        return None

    n_layers = pick(".block_count")
    n_kv = pick(".attention.head_count_kv")
    n_q = pick(".attention.head_count")
    fa_interval = pick(".full_attention_interval")

    out = {
        "family": details.get("family"),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
        "architecture": info.get("general.architecture"),
        "num_layers": n_layers,
        "num_q_heads": n_q,
        "num_kv_heads": n_kv,
        "head_dim_k": pick(".attention.key_length"),
        "head_dim_v": pick(".attention.value_length"),
        "max_position_embeddings": pick(".context_length"),
        "full_attention_interval": fa_interval,
    }
    if isinstance(n_q, int) and isinstance(n_kv, int) and n_kv:
        out["gqa_ratio"] = f"{n_q // n_kv}:1"
    # Nur bei Hybridmodellen aussagekraeftig: wie viele Layer ueberhaupt KV halten.
    if isinstance(n_layers, int) and isinstance(fa_interval, int) and fa_interval:
        out["kv_bearing_layers"] = n_layers // fa_interval
    return out


def kv_bytes_per_token(mi: dict, kv_type: str) -> Optional[float]:
    """Analytische KV-Groesse je Token, sofern die Architekturangaben reichen.

    Dient als Gegenprobe zum gemessenen VRAM-Peak: weicht beides stark ab,
    stimmt die Annahme ueber die Zahl der KV-tragenden Layer nicht.
    """
    layers = mi.get("kv_bearing_layers") or mi.get("num_layers")
    kv_heads = mi.get("num_kv_heads")
    dim_k = mi.get("head_dim_k")
    dim_v = mi.get("head_dim_v") or dim_k
    if not all(isinstance(v, int) for v in (layers, kv_heads, dim_k, dim_v)):
        return None
    # llama.cpp-Blockformate: q8_0 = 32 Werte je 34 Byte, q4_0 = 32 je 18 Byte.
    per_elem = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}.get(kv_type)
    if per_elem is None:
        return None
    return layers * kv_heads * (dim_k + dim_v) * per_elem


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="KV-Cache- und Gewichts-Quantisierungs-Benchmark auf Ollama/llama.cpp",
    )
    p.add_argument("--model", required=True, help="Ollama-Tag, z.B. qwen3.8:27b")
    p.add_argument(
        "--kv-cache-type",
        required=True,
        choices=["f16", "q8_0", "q4_0"],
        help="Nur Beschriftung -- wirksam gesetzt wird der Typ ueber OLLAMA_KV_CACHE_TYPE "
             "im Container (siehe run_matrix.sh).",
    )
    p.add_argument("--host", default="http://127.0.0.1:11435",
                   help="Bench-Instanz, NICHT die Produktions-Instanz auf 11434")
    p.add_argument("--contexts", type=int, nargs="+", required=True)
    p.add_argument("--needle-depths", type=float, nargs="+", default=DEFAULT_DEPTHS)
    p.add_argument("--decode-tokens", type=int, default=128)
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--measure-runs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="../../results/ollama_probe/raw")
    p.add_argument("--container", default="ollama-bench",
                   help="Container, dessen llama.cpp-Log fuer die Pufferwerte gelesen wird")
    p.add_argument("--tag", default=None, help="Zusatzkennung im Dateinamen, z.B. trackB")
    return p


def main() -> int:
    args = build_parser().parse_args()

    # Schutz gegen versehentliches Messen auf der Produktionsinstanz: dort
    # stehen die Env-Variablen fest, ein Lauf wuerde den Chat-Dienst blockieren
    # und trotzdem die falsche KV-Konfiguration messen.
    if ":11434" in args.host:
        print(
            "ABBRUCH: 11434 ist die Produktions-Ollama-Instanz. Die Bench-Instanz "
            "laeuft auf 11435 (siehe scripts/ollama/compose.bench.yml).",
            file=sys.stderr,
        )
        return 2

    try:
        api_get(args.host, "/api/tags", timeout=10)
    except Exception as exc:
        print(f"ABBRUCH: Bench-Instanz unter {args.host} nicht erreichbar ({exc}).",
              file=sys.stderr)
        return 2

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", args.model).strip("_").lower()
    parts = [slug, f"kv{args.kv_cache_type}"]
    if args.tag:
        parts.append(args.tag)
    experiment_id = "_".join(parts + [ts])

    print("=" * 72)
    print(f"Modell:       {args.model}")
    print(f"KV-Cache-Typ: {args.kv_cache_type}")
    print(f"Kontexte:     {args.contexts}")
    print(f"Host:         {args.host}")
    print("=" * 72)

    mi = model_info(args.host, args.model)
    print(f"Architektur:  {mi.get('architecture')} | Layer {mi.get('num_layers')} "
          f"| KV-Heads {mi.get('num_kv_heads')} | GQA {mi.get('gqa_ratio')}")
    if mi.get("kv_bearing_layers"):
        print(f"KV-tragende Layer: {mi['kv_bearing_layers']} von {mi.get('num_layers')} "
              f"(full_attention_interval={mi.get('full_attention_interval')})")

    bpt = kv_bytes_per_token(mi, args.kv_cache_type)
    if bpt:
        print(f"KV analytisch: {bpt/1024:.1f} KiB/Token "
              f"-> {bpt*max(args.contexts)/2**30:.2f} GiB bei {max(args.contexts)} Tokens")

    print("Kalibriere Zeichen-je-Token ...", flush=True)
    cpt = calibrate_chars_per_token(args.host, args.model, min(args.contexts))
    print(f"  {cpt:.3f} Zeichen/Token")

    measurements = []
    for ctx in args.contexts:
        # Vor jeder Zelle entladen, damit die VRAM-Baseline den freien Zustand
        # misst und der Peak nicht vom vorigen Kontext geerbt wird.
        unload(args.host, args.model)
        try:
            m = measure_context(
                args.host, args.model, ctx, cpt, args.needle_depths,
                args.decode_tokens, args.warmup_runs, args.measure_runs, args.seed,
                args.container, args.kv_cache_type,
            )
            measurements.append(m)
            b = m.get("buffers", {})
            print(f"  -> prefill {m['prefill_tokens_per_sec']} tok/s | "
                  f"decode {m['decode_tokens_per_sec']} tok/s | "
                  f"KV {b.get('kv_mib', 0):.0f} MiB | "
                  f"GPU gesamt {b.get('total_gpu_mib', 0):.0f} MiB | "
                  f"needle {m['needle']['successes']}/{m['needle']['trials']}")
            q = m.get("quality", {})
            mn = q.get("multi_needle", {})
            vb = q.get("verbatim", {})
            print(f"     multi-needle {mn.get('hits', '?')}/{mn.get('facts', '?')} | "
                  f"verbatim sim={vb.get('similarity', '?')} exact={vb.get('exact', '?')}")
        except Exception as exc:
            # OOM und Fenster-Ueberschreitungen sind selbst ein Messergebnis:
            # sie markieren die Grenze des Budgets und gehoeren in die Front.
            print(f"  -> FEHLER bei ctx={ctx}: {exc}", file=sys.stderr)
            measurements.append({
                "context_len": ctx,
                "error": str(exc)[:400],
                "error_type": type(exc).__name__,
            })
        unload(args.host, args.model)

    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "model_config": mi,
        "runtime": {
            "engine": "ollama/llama.cpp",
            "host": args.host,
            "weight_quant": mi.get("quantization_level"),
            "note": "Gewichte GGUF-quantisiert, nicht FP16 wie in der Arbeit; "
                    "KV-Quantisierung durch llama.cpp, nicht quanto/HQQ.",
        },
        "kv_quant": {
            "enabled": args.kv_cache_type != "f16",
            "kv_cache_type": args.kv_cache_type,
            "backend": "llama.cpp",
            "analytic_bytes_per_token": round(bpt, 1) if bpt else None,
        },
        "hardware": gpu_info(),
        "config": {
            "seed": args.seed,
            "warmup_runs": args.warmup_runs,
            "measure_runs": args.measure_runs,
            "decode_tokens": args.decode_tokens,
            "needle_depths": args.needle_depths,
            "chars_per_token": round(cpt, 4),
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
