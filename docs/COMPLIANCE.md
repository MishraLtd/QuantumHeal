# Compliance & Data Governance Note

> **Not legal advice.** This is an engineering-side summary of design intent.
> Have qualified counsel and your institution's data-protection officer review it
> before any real patient data is handled. Regulations and rules change; verify
> current text and notified rules.

## Current status of the prototype
- Tabular arm uses a **public** dataset; imaging arm uses **synthetic** images.
  **No personal data is processed today.**
- The API is built so that adding real data does not require re-architecting privacy controls.

## Design principles (mapped to what exists in the repo)
| Principle | India DPDP Act, 2023 (concept) | HIPAA (US, concept) | In this codebase |
|---|---|---|---|
| Purpose limitation & consent | Process personal data for a specified purpose with consent / permitted grounds | Minimum necessary use | Purpose stated in `/model-info`; consent capture is **not** implemented (deployer's job) |
| Data minimisation | Collect only what's needed | Minimum necessary | Model needs 30 biomarkers and/or one image; no demographics collected |
| No payload retention | Storage limitation | Safeguards on ePHI | `api.py` does not persist request bodies |
| Pseudonymised audit trail | Accountability of the Data Fiduciary | Audit controls | Audit log stores only salted-hash patient ID, time, modalities, outcome |
| Access control | Reasonable security safeguards | Access control / authentication | Optional `X-API-Key`; **production needs OAuth2/mTLS + RBAC** |
| Transport security | Reasonable security safeguards | Transmission security | Deploy behind TLS (not handled in-app) |
| Breach handling | Notification duties to Board / individuals | Breach Notification Rule | Process + tooling **not implemented** |
| Children's data | Additional protections (verifiable guardian consent) | — | Out of scope of prototype; must be addressed before paediatric use |
| Cross-border transfer | Government-notified restrictions may apply | BAAs for vendors | Design keeps data on-prem (see federated doc); cloud quantum hardware means data leaves the site → **encode/reduce on-prem, send only angle-encoded features, and get legal sign-off** |

## Quantum-cloud caveat
Running on IBM/AWS hardware sends circuit inputs (encoded feature angles) to a
third party. Even reduced features can be personal data. Prefer simulators on-prem
for real patient data until a data-processing agreement is in place.

## Medical-device / clinical-use considerations
Diagnostic decision-support software may be regulated as a medical device
(e.g. CDSCO in India, FDA in the US). Prototype outputs carry a "research
prototype, not a diagnosis" disclaimer. Clinical validation on real, representative,
prospectively collected data — with bias/subgroup analysis — is required before any claim.

## Interoperability
India's ABDM ecosystem and HL7 FHIR are the natural integration targets; the current
JSON API is a simple stand-in. A FHIR `RiskAssessment` response mapping is a suggested next step.

## Pre-production checklist
- [ ] DPO / legal review; lawful basis + consent workflow
- [ ] DPIA (data protection impact assessment)
- [ ] Authn/z (OAuth2, RBAC), TLS everywhere, secrets management (rotate `QHS_AUDIT_SALT`)
- [ ] Log retention policy, breach playbook
- [ ] De-identification standard for any training export
- [ ] Clinical validation + regulatory pathway decision
