# Quantum-Assisted Synthetic Data Augmentation (design — not built)

**Status:** scoped, not implemented. Included because medical datasets are small and
privacy-restricted, so synthetic augmentation is a natural roadmap item.

## Honest framing
Quantum GANs on today's simulators/NISQ devices generate tiny distributions
(a handful of qubits → a few features). They are a research direction, **not** an
established route to better medical data than classical generators.

## Realistic scope
- Target: **tabular biomarker embeddings** (the 4-dim PCA space the VQC already consumes),
  not raw images. Generating 64×64 images on 4 qubits is not feasible.
- Generator: parameterised circuit (same ansatz family as `build_vqc_arm`) sampling
  angle vectors; discriminator: small classical MLP (hybrid QGAN).
- Use: augment minority class (malignant) in low-data regimes; evaluate by
  *train-on-synthetic, test-on-real* accuracy vs. a classical baseline (SMOTE / CTGAN).

## Success criteria (decide before building)
1. Beats or matches SMOTE on held-out real data.
2. Privacy check: nearest-neighbour distance to real training records shows no memorisation.
3. If it does not beat classical baselines, report that plainly.
