# Results

**Nothing to report yet — honestly.** Metrics land here only when produced by real
training runs on the real dataset, with the commit hash that produced them.

Planned tables:

- **D2–4** — per-category IoU and pixel precision/recall on the held-out test split.
- **D5** — conformal: λ̂ per category, held-out FNR vs α (=0.1), mean predicted mask
  fraction, and the FNR-vs-λ risk curve.
- **D6** — ONNX parity (max |Δ|) and CPU inference latency via onnxruntime.

House rule, same as the sibling repos: the headline is whatever the data says —
including "the guarantee held but the masks are uselessly large", if that is what
happens at small n.
