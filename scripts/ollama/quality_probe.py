#!/usr/bin/env python3
"""Qualitaetssonden fuer den Ollama-Messpfad.

Needle-Retrieval allein zeigt nur Zusammenbrueche: der Code wird gefunden oder
nicht. Fuer die Frage, ob ein Arbeitspunkt fuer Coding taugt, ist das zu grob --
dort aeussert sich Schaden als schleichende Verschlechterung, lange bevor das
Modell die Information ganz verliert. Die Arbeit loest das ueber Perplexitaet;
die ist ueber die Ollama-API nicht zugaenglich, weil dort keine Logprobs
herauskommen.

Ersatz sind drei Sonden, die alle rein ueber die Chat-API laufen:

1. ``multi_needle`` -- fuenf verschiedene Codes in unterschiedlichen Tiefen,
   gemeinsam abgefragt. Anders als der Einzel-Needle misst das, ob das Modell
   mehrere verstreute Fakten gleichzeitig halten kann; Degradation zeigt sich
   hier als Teiltreffer statt als Totalausfall.

2. ``verbatim`` -- ein Codeblock im Kontext, der woertlich reproduziert werden
   soll. Objektiv scorebar ueber Zeichenaehnlichkeit und damit feinkoernig:
   ein einzelnes falsches Zeichen ist messbar, ohne dass die Antwort unbrauchbar
   waere. Das kommt der Coding-Frage am naechsten.

3. ``greedy_sample`` -- bei temperature 0 und top_k 1 erzeugter Text, im
   Ergebnis abgelegt. Bei identischem Prompt und identischen Gewichten ist jede
   Abweichung zwischen zwei KV-Konfigurationen allein durch die Quantisierung
   verursacht. ``aggregate_ollama.py`` vergleicht das nachtraeglich gegen die
   f16-Referenz; hier wird nur eingesammelt.
"""

import difflib
import re
from typing import Callable, Dict, List

# Bewusst mit Zahlen, Sonderzeichen und Einrueckung: das sind die Stellen, an
# denen ein beschaedigter Cache zuerst danebengreift.
CODE_SNIPPET = """def rescale_tile(tile, factor=0.375, clamp=(-2048, 2048)):
    lo, hi = clamp
    scaled = [min(max(int(v * factor), lo), hi) for v in tile]
    checksum = sum(scaled) % 65521
    return {"tile": scaled, "checksum": checksum, "factor": factor}"""

VERBATIM_MARKER = "### FUNKTION-A7 ###"

MULTI_NEEDLE_DEPTHS = [0.08, 0.28, 0.48, 0.68, 0.88]
MULTI_NEEDLE_FACTS = [
    ("Rechenzentrum Berlin", "Zugangsnummer"),
    ("Ersatzteillager Hamburg", "Regalnummer"),
    ("Kuehlkreislauf Muenchen", "Ventilnummer"),
    ("Notstromaggregat Leipzig", "Pruefnummer"),
    ("Serverraum Dresden", "Schrankummer"),
]


def _insert_at_depth(body: str, text: str, depth: float) -> str:
    """Fuegt ``text`` an relativer Tiefe ein, auf der naechsten Wortgrenze."""
    cut = int(len(body) * depth)
    space = body.find(" ", cut)
    cut = space if space != -1 else cut
    return body[:cut] + "\n\n" + text + "\n\n" + body[cut:]


def multi_needle(
    generate: Callable[..., dict],
    context_len: int,
    filler_body: str,
    seed: int,
) -> dict:
    """Fuenf Fakten in verschiedenen Tiefen, gemeinsam abgefragt."""
    body = filler_body
    codes = {}
    # Von hinten nach vorne einfuegen, damit fruehere Tiefen nicht verschoben
    # werden.
    for i, (ort, art) in reversed(list(enumerate(MULTI_NEEDLE_FACTS))):
        code = f"{(context_len // 1024) * 1000 + i * 7 + 3:07d}"
        codes[ort] = code
        body = _insert_at_depth(
            body, f"Die {art} fuer {ort} lautet {code}.", MULTI_NEEDLE_DEPTHS[i]
        )

    frage = (
        "\n\nFrage: Nenne fuer jeden der folgenden Orte die im Text genannte "
        "Nummer, je Zeile im Format ORT=NUMMER, ohne weiteren Text:\n"
        + "\n".join(ort for ort, _ in MULTI_NEEDLE_FACTS)
    )

    resp = generate(prompt=body + frage, num_predict=160, seed=seed, think=False)
    answer = resp.get("response", "") or ""

    hits = {}
    for ort, code in codes.items():
        # Zeile zum Ort suchen und darin die Ziffernfolge pruefen.
        line = next(
            (l for l in answer.splitlines() if ort.split()[0].lower() in l.lower()), ""
        )
        hits[ort] = code in re.sub(r"[^0-9]", "", line)

    return {
        "facts": len(codes),
        "hits": sum(hits.values()),
        "hit_rate": round(sum(hits.values()) / len(codes), 4),
        "per_fact": hits,
        "answer": answer.strip()[:400],
    }


def verbatim_reproduction(
    generate: Callable[..., dict],
    filler_body: str,
    seed: int,
) -> dict:
    """Codeblock im Kontext, der woertlich reproduziert werden soll."""
    block = f"{VERBATIM_MARKER}\n{CODE_SNIPPET}\n### ENDE ###"
    body = _insert_at_depth(filler_body, block, 0.5)

    frage = (
        f"\n\nAufgabe: Gib den Codeblock, der im Text unter der Markierung "
        f"{VERBATIM_MARKER} steht, exakt und unveraendert wieder. "
        f"Keine Erklaerung, keine Code-Fences, nur den Code."
    )

    resp = generate(prompt=body + frage, num_predict=256, seed=seed, think=False)
    answer = (resp.get("response", "") or "").strip()

    # Fences entfernen, falls das Modell sie trotz Anweisung setzt -- die sind
    # ein Formatierungsdetail und nicht der zu messende Schaden.
    cleaned = re.sub(r"^```[a-zA-Z]*\n|```$", "", answer, flags=re.MULTILINE).strip()

    ratio = difflib.SequenceMatcher(None, CODE_SNIPPET, cleaned).ratio()
    return {
        "exact": cleaned == CODE_SNIPPET,
        "similarity": round(ratio, 4),
        "checksum_correct": "65521" in cleaned,
        "factor_correct": "0.375" in cleaned,
        "clamp_correct": "2048" in cleaned,
        "answer_len": len(cleaned),
        "answer": cleaned[:400],
    }


GREEDY_TASKS = [
    "Fasse den obigen Text in genau drei Saetzen zusammen.",
    "Schreibe eine Python-Funktion, die aus dem obigen Text alle genannten "
    "Nummern extrahiert und als sortierte Liste zurueckgibt. Nur Code.",
]


def greedy_samples(
    generate: Callable[..., dict],
    filler_body: str,
    seed: int,
) -> List[dict]:
    """Greedy erzeugte Ausgaben zum spaeteren Vergleich gegen die f16-Referenz.

    Nicht hier bewertet: der Vergleich braucht zwei Konfigurationen und passiert
    deshalb in der Aggregation.
    """
    out = []
    for i, task in enumerate(GREEDY_TASKS):
        resp = generate(
            prompt=filler_body + "\n\n" + task, num_predict=200, seed=seed, think=False
        )
        text = (resp.get("response", "") or "").strip()
        out.append({
            "task_index": i,
            "task": task[:80],
            "output": text,
            "eval_count": resp.get("eval_count"),
        })
    return out


def run_all(
    generate: Callable[..., dict],
    context_len: int,
    filler_body: str,
    seed: int,
) -> Dict:
    """Alle drei Sonden. ``generate`` muss prompt/num_predict/seed/think nehmen."""
    result = {}
    for name, fn in (
        ("multi_needle", lambda: multi_needle(generate, context_len, filler_body, seed)),
        ("verbatim", lambda: verbatim_reproduction(generate, filler_body, seed)),
        ("greedy_samples", lambda: greedy_samples(generate, filler_body, seed)),
    ):
        try:
            result[name] = fn()
        except Exception as exc:
            # Eine gescheiterte Sonde darf die uebrigen nicht mitreissen.
            result[name] = {"error": str(exc)[:300], "error_type": type(exc).__name__}
    return result
