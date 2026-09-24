# Ueberholte Messungen: UD-Modelle ohne Draft und ohne Vision

Diese Laeufe stammen aus dem ersten Nachtlauf (2026-08-27, Tag `p1`) und messen
die UD-Quants **unfair**. Ihr Modelfile enthielt nur `FROM <gguf>`:

- kein `PARAMETER draft_num_predict` -- die Library-Variante faehrt damit
  spekulatives Decoding ueber den MTP-Kopf (~3,6 akzeptierte Token je Draft),
  die UD-Modelle liefen ohne. Der gemessene Decode-Rueckstand von 40--60 % ist
  dadurch ein Artefakt des Imports, keine Eigenschaft der Quantisierung.
- kein Vision-Projektor -- senkt die VRAM-Belegung um rund 1,1 GB gegenueber
  der vergleichbar ausgestatteten Library-Variante.
- kein `RENDERER`/`PARSER`.

Der MTP-Kopf steckt in beiden GGUFs: 866 Tensoren, identisch, einschliesslich
`blk.64.nextn.*`. Er war beim UD-Import nur nie eingeschaltet.

**Weiterhin gueltig** sind die Qualitaetswerte (Needle, Multi-Needle, Verbatim --
die Sonden liefen unveraendert) und die KV-Puffergroessen. **Ungueltig** sind
Decode-Durchsatz und der VRAM-Vergleich gegen die Library-Variante.

Ersetzt durch die Laeufe mit Tag `fix` und `fix256` auf `qwen3.8-udv:*`.
