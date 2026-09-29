# Federated Deployment Architecture (design — not built)

**Status:** design document. Nothing here is implemented; it describes how the
current platform would be deployed across hospitals that cannot pool patient data.

## Goal
Each hospital trains on its own patients; only model parameters leave the site.
No biomarkers, images or patient IDs are ever transmitted.

## Why this platform suits it
The quantum arms are tiny: a 4-qubit, 3-layer VQC has **12 weights + 1 bias**.
The CNN embedding is a few thousand parameters. Update payloads are a few KB,
so bandwidth is a non-issue and updates are easy to inspect and audit.

## Topology
```
 Hospital A (on-prem)          Hospital B (on-prem)          Hospital C
 ┌───────────────────┐        ┌───────────────────┐        ┌────────────┐
 │ local data (EHR,  │        │ local data        │        │   ...      │
 │ imaging archive)  │        │                   │        │            │
 │ local trainer     │        │ local trainer     │        │            │
 │ (tab VQC, CNN,    │        │                   │        │            │
 │  img VQC)         │        │                   │        │            │
 └────────┬──────────┘        └────────┬──────────┘        └─────┬──────┘
          │ weight deltas only (TLS, signed)                      │
          └───────────────►  Aggregation server  ◄────────────────┘
                              - FedAvg of VQC + CNN weights
                              - secure aggregation / clipping + DP noise
                              - global model registry (versioned)
                                       │ new global weights
                                       ▼
                              pushed back to every site
```

## Round protocol (FedAvg)
1. Server publishes global weights `W_t`.
2. Each site runs k local epochs on its own data → `W_t^(i)`; sends `ΔW = W_t^(i) − W_t`
   plus its sample count `n_i` (nothing else).
3. Server clips each `ΔW` (L2 norm bound), adds calibrated Gaussian noise
   (differential privacy), averages weighted by `n_i`, publishes `W_{t+1}`.
4. Repeat; hold-out evaluation happens **at each site** and only aggregate metrics are shared.

## Quantum-specific notes
- VQC weights are angles; FedAvg over angles is valid as parameter averaging but
  angle wrap-around (±π) should be handled (average on the circle or re-center).
- Sites may use different backends (simulator vs hardware). Keep a shared
  circuit definition (`build_vqc_arm`) and versioned encoding/scaling so
  embeddings are comparable — **preprocessing statistics (scaler/PCA) must be
  federated too** (share means/covariances with noise, or fix a public reference).
- Non-IID data across hospitals (different scanners, populations) is the main
  scientific risk; plan per-site fine-tuning of the fusion weights.

## Integration with existing code
| Component | Change needed |
|---|---|
| `build_vqc_arm().train` | expose `init_weights` + return delta; trivial |
| `CNNFeaturePipeline` | expose state_dict get/set |
| `api.py` | load global weights from registry at startup |
| New | aggregation service (e.g. Flower or a small FastAPI app), model registry, site enrollment/auth |

## Suggested tooling
Flower (`flwr`) for orchestration; Opacus-style clipping/noise for DP; mTLS between sites.

## Open questions / risks
Privacy of weight updates (membership inference) even with DP; regulatory status of
model sharing between institutions; governance of who owns the global model.
