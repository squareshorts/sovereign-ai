# SPST Reference Implementation Results

## Scientifically Defensible Conclusion
This reference implementation demonstrates that the Sovereign Purpose-Specific Tooling (SPST) architecture and its conformance machinery can be implemented and automatically tested. The test harness successfully enforces exportability, reconnection constraints, authorization boundaries, provenance tracking, and performance envelopes using deterministic mock fixtures.

**Limitations:** The tests use deterministic fixtures. The results DO NOT establish behavioral portability across real foundation models or organizational providers. The scientifically permitted conclusion is limited to demonstrating that the SPST architecture and its conformance machinery can be implemented and evaluated automatically.

## Suitability for Supplementary Information
The provided package, including its isolated conformance testing mechanisms, zero-dependency nature (standard library only), and deterministic fixtures, is fully self-contained. It serves as robust technical evidence suitable for Supplementary Information to back the architectural claims of the main manuscript without making unsubstantiated claims about real-world foundation models.

---

## Clean Package Inventory
The final package has been purged of cache directories (`__pycache__` and `.pyc`), credentials, `.claude` files, or any environment-specific junk (verified prior to packaging).
The results data (`spst_results.csv`, `summary_results.json`) and the generated exported packages have been successfully included in `results/`.

---

## Test Suite Results
* **Tests Executed:** 130
* **Passed:** 130
* **Failed:** 0
* **Skipped/Errored:** 0

---

## SPST Harness Execution Results (SPST-1 to SPST-6)

### SPST-1: Exportability
* **Files exported:** 8 (Institutional files only)
* **Package size:** 36,780 bytes
* **Adapter files included:** 0
* **Manifest loadable:** True
* **`reconstructed_module_isolated`:** True
* **Clean reconstruction successful:** True
* **Regression output equivalent:** True
* **Result:** PASS
*(Note: SPST-1 PASS is confirmed because the reconstructed `workflow.task` was demonstrably loaded from the reconstruction/export path rather than the original repository.)*

### SPST-2: Reconnection
* **Institutional files changed:** 0
* **Institutional hash equal:** True
* **Adapter-specific LOC:** 174
* **Result:** PASS

### SPST-3: Authorization Preservation
* **Authorization test cases:** 5
* **Fixtures tested:** 4
* **Total negative tests executed:** 20
* **All prohibited operations blocked:** True
* **All resource gates blocked:** True
* **Prohibited operations actually executed:** 0
* **Result:** PASS

### SPST-4: Provenance Preservation
* **Total Records Evaluated:** 40
* **Complete Records:** 30/40 (75.00%)
* **Conformant Fixture:** Complete provenance validated
* **Schema-Failure Fixture:** Complete provenance validated
* **Performance-Failure Fixture:** Complete provenance validated
* **Provenance-Failure Fixture:** Incomplete provenance (Correctly detected by the machinery)
* **Result:** PASS

### SPST-5: Performance Acceptance Machinery
*(NOTE: Substantive performance portability was NOT evaluated because no independent real inference models were available.)*

**Conformant Fixture**
* **Schema valid rate:** 100.00%
* **Discrepancy TP/FP/FN/TN:** 35/0/0/20
* **Discrepancy sensitivity:** 100.00%
* **Discrepancy precision:** 100.00%
* **Human-review sensitivity:** 100.00%
* **Critical error rate:** 0.00%
* **Meets acceptance envelope:** True (All required thresholds enforced)

**Schema-Failure Fixture**
* **Schema valid rate:** 0.00%
* **Discrepancy TP/FP/FN/TN:** 0/0/35/0
* **Discrepancy sensitivity:** 0.00%
* **Human-review sensitivity:** 0.00%
* **Critical error rate:** 0.00%
* **Meets acceptance envelope:** False

**Performance-Failure Fixture**
* **Schema valid rate:** 100.00%
* **Discrepancy TP/FP/FN/TN:** 0/0/35/20
* **Discrepancy sensitivity:** 0.00%
* **Human-review sensitivity:** 0.00%
* **Critical error rate:** 0.00%
* **Meets acceptance envelope:** False

**Provenance-Failure Fixture**
* **Schema valid rate:** 100.00%
* **Discrepancy TP/FP/FN/TN:** 35/0/0/20
* **Discrepancy sensitivity:** 100.00%
* **Discrepancy precision:** 100.00%
* **Human-review sensitivity:** 100.00%
* **Critical error rate:** 0.00%
* **Meets acceptance envelope:** True

* **Result:** PASS (acceptance-envelope machinery functions correctly)

### SPST-6: Reversibility
* **Active configuration A loaded:** Verified
* **Switch to B confirmed:** Verified
* **Configuration restored to A:** Verified
* **Restored config == baseline:** True
* **Validated config matches manifest:** True
* **0 institutional files changed:** True
* **Regression outputs A-before == A-after:** True
* **Provenance continuity:** True
* **Result:** PASS
