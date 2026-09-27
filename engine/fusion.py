"""
QuantumHealth Sentinel — Late-Fusion Layer  (Phase 1: Multimodal Expansion)
=============================================================================
Combines per-modality VQC outputs (currently: tabular + imaging) into a
single fused prediction.

Implements:
  - Late fusion: weighted average of each arm's P(disease) output.
    Chosen over early fusion (concatenate-then-encode) because it keeps
    each modality's quantum circuit small and independently trainable/
    explainable, and is the safer, demo-able choice under hackathon time
    constraints (see design notes shared with this phase).
  - Missing-modality graceful degradation: if a patient has no imaging
    (or, in future, no tabular panel), fuse using only the modality(ies)
    that ARE present, with weights renormalized — the platform never
    hard-fails on partial real-world intake data.
  - Per-modality attribution: how much each present modality contributed
    to the fused call, surfaced in the dashboard's explainability tab.

Learned/attention-based fusion (per-patient modality weighting) is
scoped as a Phase 2 upgrade — see README delivery table, item 15/19.
"""

import numpy as np


class LateFusion:
    def __init__(self, weights=None):
        # Fixed default weights for Phase 1 (tabular trusted slightly more,
        # since it's benchmarked against real clinical data while the
        # imaging arm currently trains on synthetic data — see imaging_arm.py).
        # Phase 2 fits these against a validation set instead of hardcoding.
        self.weights = weights or {"tabular": 0.6, "imaging": 0.4}

    def fuse(self, tabular_prob=None, imaging_prob=None):
        """Each *_prob is an array of per-patient P(disease) from that arm,
        or None if that modality was not collected for this batch.

        Returns (fused_prob, fused_pred, contributions) where
        `contributions` is a dict of the normalized weight each present
        modality received in the fused score (for explainability).
        """
        present = {}
        if tabular_prob is not None:
            present["tabular"] = np.asarray(tabular_prob, dtype=float)
        if imaging_prob is not None:
            present["imaging"] = np.asarray(imaging_prob, dtype=float)
        if not present:
            raise ValueError("At least one modality's probability output is required.")

        total_w = sum(self.weights[m] for m in present)
        fused = sum(self.weights[m] * p for m, p in present.items()) / total_w
        contributions = {m: round(self.weights[m] / total_w, 3) for m in present}
        preds = (fused > 0.5).astype(int)
        return fused, preds, contributions
