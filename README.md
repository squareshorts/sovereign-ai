# SPST Reference Implementation

## Sovereign Provider-Switching Test — Conformance Harness

This is a **reference implementation** of the Sovereign Provider-Switching
Test (SPST). It demonstrates that workflow export, adapter-bounded
reconnection, authorization enforcement, provenance completeness,
acceptance-envelope enforcement, and rollback can be represented and
automatically tested as conformance properties.

### Important Disclaimer

This implementation uses **deterministic mock conformance fixtures**, not
real foundation models or commercial inference providers. It does NOT
establish behavioral portability across real models or organizational
providers.

## Requirements

- **Python 3.8+** (standard library only; no external dependencies)
- No API keys or credentials required

## Directory Structure

```
sovereign_ai/
├── README.md
├── workflow_manifest.json       # Institution-controlled workflow specification
├── run_spst.py                  # Main conformance harness
├── workflow/                    # Institution-controlled layer
│   ├── schemas.py               #   Input/output schema validation
│   ├── authorization.py         #   Authorization enforcement engine
│   └── task.py                  #   Reconciliation task + provenance
├── adapters/                    # Provider abstraction boundary
│   ├── base.py                  #   Base adapter interface
│   ├── conformant.py            #   Conformant fixture
│   ├── schema_failure.py        #   Schema-failure fixture
│   ├── performance_failure.py   #   Performance-failure fixture
│   └── provenance_failure.py    #   Provenance-failure fixture
├── evaluation/                  # Evaluation suite
│   ├── synthetic_data.py        #   60 ground-truth-annotated cases
│   └── metrics.py               #   TP/FP/FN/TN metric computation
├── config/                      # Deployment configurations
│   ├── deployment_a.json        #   Fixture A config
│   └── deployment_b.json        #   Fixture B config
├── tests/                       # Automated test suite
│   └── test_spst.py
└── results/                     # Generated outputs
    ├── spst_results.csv
    ├── summary_results.json
    └── export_package/
```

## How to Run

### Full conformance harness
```bash
python run_spst.py
```

### Automated test suite
```bash
python -m tests.test_spst
```

## What This Demonstrates

| SPST Component | What is tested |
|---|---|
| SPST-1: Exportability | Institutional workflow packaged without adapter state |
| SPST-2: Reconnection | Adapter switch changes 0 institutional files |
| SPST-3: Authorization | Prohibited ops detected and blocked externally |
| SPST-4: Provenance | Full audit trail validated against manifest |
| SPST-5: Performance | Acceptance-envelope machinery functions correctly |
| SPST-6: Reversibility | A→B→A rollback preserves institutional state |

## What This Does NOT Demonstrate

- Clinical safety or effectiveness
- Behavioral portability across real foundation models
- Full organizational provider switching
- Model equivalence or quality comparison
