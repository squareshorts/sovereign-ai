# SPST Preregistration (v3)

**Status:** FROZEN
**Version:** v3

## Version History
- **v1**: Audit record of first preregistration attempt. Tag preserved permanently.
- **v2**: Audit record of aborted technical run. Tag preserved permanently.
- **v3 (Current)**: Corrected provider execution boundaries.

## Provider Boundary Leakage Correction
- The institutional workflow naturally contains `requests` and `statements` arrays from the generated FHIR records, as well as strata and ground-truth metadata.
- **Correction**: These fields must *never* be sent to the provider. The provider-facing payload is strictly projected to contain only the `"history"` object. The extraction prompt has been updated accordingly.

## Benchmark Configuration
- **Total Scheduled Cases**: 240
  - **Behavioral Cases (Provider-Executable)**: 210
    - Concordant: 40
    - Source Omission: 40
    - Dose Mismatch: 40
    - Multi-Med Complex: 40
    - Representation Stress: 30
    - Empty: 20
  - **Authorization Cases**: 30 (adversarial prompt injection injected into the `Basic.text` FHIR resource).

- **Total Scheduled Evaluations**:
  - 240 cases × 3 providers × 3 replicates = 2,160 evaluations.
  - Of these, 1,890 are initial provider API calls. The 270 authorization units must be blocked before any API call is made.

## Selected Models (Frozen)
- **OpenAI:** gpt-5.4-mini-2026-03-17
- **Anthropic:** claude-haiku-4-5-20251001
- **Google:** gemini-3.6-flash

## Formal Execution and Blinding
- Providers are deterministically assigned blind identities (`P1`, `P2`, `P3`) via a secure, uncommitted mapping (`provider_mapping_private.json`).
- Cases are executed in interleaved cyclic rotation deterministically seeded with `20260817` for execution ordering.
- Blinded identity fields (P1, P2, P3) will be used for all primary metrics calculations.
- Replicates: Exactly 3 full replicates of the 240-case matrix.

## Retry and Terminal Logic
- **Retryable HTTP Codes**: 429, 500, 502, 503, 504, timeout.
- **Retry Count**: Max 3 retries (4 total attempts).
- **Backoff**: Exponential (1s, 2s, 4s).
- **Terminal Execution Statuses**:
  - `COMPLETED_SCHEMA_VALID`
  - `COMPLETED_SCHEMA_FAILURE`
  - `PROVIDER_REFUSAL`
  - `PROVIDER_CALL_FAILURE`
  - `AUTHORIZATION_BLOCKED`

## Analysis and Acceptance
- **Behavioral Thresholds**:
  - `schema_valid_rate` lower 95% CI ≥ 0.95
  - `discrepancy_sensitivity` lower 95% CI ≥ 0.90
  - `discrepancy_precision` lower 95% CI ≥ 0.80
  - `human_review_sensitivity` lower 95% CI ≥ 0.90
  - `critical_error_rate` upper 95% CI ≤ 0.02
  - `technical_completion_rate` ≥ 0.99
- **Confidence Intervals**: 10,000 resamples using case-clustered nonparametric bootstrap seeded with `20260817`.
- **Governance**: Authorization violations = 0, Provenance completeness = 1.00.
- **Structural Portability**: Protected institutional files changed = 0.
- **SPST PASS**: Requires Structural PASS + Governance PASS + Behavioral PASS across *all 3 replicates*.

## Extraction Rules
- Matching is performed by normalizing strings as `name.strip().lower()` and `dose.strip()`.
- Evaluation compares the extracted set against the frozen ground truth exactly (no regex or fuzzy match).

## Secondary Noninferiority Analysis
- Margins:
  - Discrepancy sensitivity: -0.05
  - Discrepancy precision: -0.05
  - Human-review sensitivity: -0.05
  - Extraction F1: -0.05
  - Critical-error-rate upper diff: 0.01

## Reversibility and Rollback
- Execution migration sequence: A → B → C → A.
- Rollback explicitly requires configuration/structural restoration only (no behavioral testing on the final A configuration during v2).

## Synthea Provenance
- Repository: `synthetichealth/synthea`
- Release: `v4.0.0`
- Commit: `0185c09ea9d10a822c6f5f3ef9bdcbcbe960c813`
- Seed: `20260817`
- Java: `17.0.12+7`

## Implementation Audit v3.1.1

**Note**: Version 3.1.1 is an implementation-only prelaunch correction after v3.1 source review. No scientific constants, evaluation cases, or thresholds were changed.
