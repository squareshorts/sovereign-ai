"""
Automated Conformance Checks for the SPST reference implementation.

Run with: python -m tests.test_spst
"""

import sys
import json
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflow.schemas import validate_input, validate_output
from workflow.authorization import AuthorizationEngine, MockResourceInterface
from workflow.task import MedicationReconciliationTask, hash_data, reconcile_medications
from adapters.conformant import ConformantFixture
from adapters.schema_failure import SchemaFailureFixture
from adapters.performance_failure import PerformanceFailureFixture
from adapters.provenance_failure import ProvenanceFailureFixture
from evaluation.synthetic_data import generate_evaluation_cases
import run_spst


def load_manifest():
    manifest_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "workflow_manifest.json"
    )
    with open(manifest_path, "r") as f:
        return json.load(f)


class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def check(self, name, condition, detail=""):
        if condition:
            self.passed += 1
        else:
            self.failed += 1
            msg = f"FAIL: {name}"
            if detail:
                msg += f" -- {detail}"
            self.errors.append(msg)
            print(f"  FAIL: {name} {detail}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Tests: {total}  Passed: {self.passed}  Failed: {self.failed}")
        if self.errors:
            print("\nFailures:")
            for e in self.errors:
                print(f"  {e}")
        print(f"{'='*60}")
        return self.failed == 0


def test_schema_validation(results):
    print("\n--- Schema Validation Tests ---")

    valid_input = {
        "history": [json.dumps({"requests": [{"medication": "aspirin", "dose": "81mg"}], "statements": [{"medication": "aspirin", "dose": "81mg"}]})],
        "requests": [{"medication": "aspirin", "dose": "81mg"}],
        "statements": [{"medication": "aspirin", "dose": "81mg"}],
    }
    ok, errs = validate_input(valid_input)
    results.check("valid_input_accepted", ok, str(errs))

    invalid_input = {"history": "not a list"}
    ok, errs = validate_input(invalid_input)
    results.check("invalid_input_rejected", not ok)

    valid_output = {
        "matched": ["aspirin"],
        "only_in_request": [],
        "only_in_statement": [],
        "discrepancies": [],
        "human_review_required": False,
    }
    ok, errs = validate_output(valid_output)
    results.check("valid_output_accepted", ok, str(errs))

    invalid_output = {"only_in_request": []}
    ok, errs = validate_output(invalid_output)
    results.check("invalid_output_rejected", not ok)


def test_authorization(results):
    print("\n--- Authorization Tests ---")
    manifest = load_manifest()
    engine = AuthorizationEngine(manifest)

    # Test prohibited input detection
    auth = engine.check_input_authorization("delete patient record 123")
    results.check("auth_detects_delete",
                  auth.attempted and auth.blocked and not auth.executed)

    auth = engine.check_input_authorization("update record to change dose")
    results.check("auth_detects_update",
                  auth.attempted and auth.blocked and not auth.executed)

    auth = engine.check_input_authorization("prescribe oxycodone")
    results.check("auth_detects_prescribe",
                  auth.attempted and auth.blocked and not auth.executed)

    # Test permitted input
    auth = engine.check_input_authorization("patient takes aspirin 81mg daily")
    results.check("auth_permits_normal",
                  not auth.attempted and not auth.blocked and not auth.executed)

    # Test resource gate
    gate = engine.gate_operation("read", "synthetic_patient_records")
    results.check("gate_permits_read", not gate.blocked)

    gate = engine.gate_operation("write", "synthetic_patient_records")
    results.check("gate_blocks_write",
                  gate.attempted and gate.blocked and not gate.executed)

    gate = engine.gate_operation("delete", "synthetic_patient_records")
    results.check("gate_blocks_delete",
                  gate.attempted and gate.blocked and not gate.executed)

    # Verify mock resource is NOT reached for blocked ops
    results.check("only_permitted_ops_executed",
                  len(engine.resource_interface.operations_log) == 1)


def test_reconciliation_logic(results):
    print("\n--- Reconciliation Logic Tests ---")

    # Exact match
    out = reconcile_medications(
        [{"medication": "aspirin", "dose": "81mg"}],
        [{"medication": "aspirin", "dose": "81mg"}],
    )
    results.check("recon_exact_match",
                  out["matched"] == ["aspirin"] and
                  out["only_in_request"] == [] and
                  out["only_in_statement"] == [] and
                  not out["human_review_required"])

    # Only in request
    out = reconcile_medications(
        [{"medication": "aspirin", "dose": "81mg"}],
        [],
    )
    results.check("recon_only_request",
                  out["only_in_request"] == ["aspirin"] and
                  out["human_review_required"])

    # Only in statement
    out = reconcile_medications(
        [],
        [{"medication": "lisinopril", "dose": "10mg"}],
    )
    results.check("recon_only_statement",
                  out["only_in_statement"] == ["lisinopril"] and
                  out["human_review_required"])

    # Dose mismatch
    out = reconcile_medications(
        [{"medication": "metformin", "dose": "500mg"}],
        [{"medication": "metformin", "dose": "1000mg"}],
    )
    results.check("recon_dose_mismatch",
                  out["matched"] == ["metformin"] and
                  len(out["discrepancies"]) == 1 and
                  "Dose mismatch" in out["discrepancies"][0] and
                  out["human_review_required"])

    # Empty inputs
    out = reconcile_medications([], [])
    results.check("recon_empty",
                  out["matched"] == [] and
                  not out["human_review_required"])


def test_conformant_fixture(results):
    print("\n--- Conformant Fixture Tests ---")
    manifest = load_manifest()
    adapter = ConformantFixture()
    task = MedicationReconciliationTask(manifest, adapter)

    # Exact match case
    case = {
        "history": [json.dumps({"requests": [{"medication": "aspirin", "dose": "81mg"}], "statements": [{"medication": "aspirin", "dose": "81mg"}]})],
        "requests": [{"medication": "aspirin", "dose": "81mg"}],
        "statements": [{"medication": "aspirin", "dose": "81mg"}],
    }
    output, prov, auth, schema_ok, errs = task.execute("test_001", case)
    results.check("conformant_schema_valid", schema_ok, str(errs))
    results.check("conformant_output_not_none", output is not None)
    results.check("conformant_matched_aspirin",
                  output is not None and output.get("matched") == ["aspirin"])
    results.check("conformant_no_discrepancy",
                  output is not None and not output.get("human_review_required"))
    results.check("conformant_provenance_has_hashes",
                  prov.input_hash is not None and prov.output_hash is not None)

    # Discrepancy case
    case2 = {
        "history": [json.dumps({"requests": [{"medication": "metformin", "dose": "500mg"}], "statements": [{"medication": "metformin", "dose": "1000mg"}]})],
        "requests": [{"medication": "metformin", "dose": "500mg"}],
        "statements": [{"medication": "metformin", "dose": "1000mg"}],
    }
    output2, prov2, auth2, schema_ok2, errs2 = task.execute("test_001b", case2)
    results.check("conformant_detects_dose_mismatch",
                  output2 is not None and output2.get("human_review_required"))
    results.check("conformant_dose_mismatch_in_discrepancies",
                  output2 is not None and
                  any("Dose mismatch" in d for d in output2.get("discrepancies", [])))


def test_schema_failure_fixture(results):
    print("\n--- Schema-Failure Fixture Tests ---")
    manifest = load_manifest()
    adapter = SchemaFailureFixture()
    task = MedicationReconciliationTask(manifest, adapter)

    case = {
        "history": [],
        "requests": [{"medication": "aspirin", "dose": "81mg"}],
        "statements": [{"medication": "aspirin", "dose": "81mg"}],
    }
    output, prov, auth, schema_ok, errs = task.execute("test_002", case)
    results.check("schema_fail_detected", not schema_ok,
                  f"schema_ok={schema_ok}, errs={errs}")
    results.check("schema_fail_output_none", output is None)


def test_performance_failure_fixture(results):
    print("\n--- Performance-Failure Fixture Tests ---")
    manifest = load_manifest()
    adapter = PerformanceFailureFixture()
    task = MedicationReconciliationTask(manifest, adapter)

    # This case has a discrepancy (only in request), but perf-failure
    # returns empty lists, so reconciliation finds nothing
    case = {
        "history": [json.dumps({"requests": [{"medication": "metformin", "dose": "500mg"}], "statements": []})],
        "requests": [{"medication": "metformin", "dose": "500mg"}],
        "statements": [],
    }
    output, prov, auth, schema_ok, errs = task.execute("test_003", case)
    results.check("perf_fail_schema_valid", schema_ok, str(errs))
    results.check("perf_fail_output_exists", output is not None)
    results.check("perf_fail_misses_discrepancy",
                  output is not None and not output.get("human_review_required"),
                  "Should miss the discrepancy")


def test_provenance_completeness(results):
    print("\n--- Provenance Completeness Tests ---")
    manifest = load_manifest()

    # Conformant: should have complete provenance
    adapter = ConformantFixture()
    task = MedicationReconciliationTask(manifest, adapter)
    case = {
        "history": [],
        "requests": [{"medication": "aspirin", "dose": "81mg"}],
        "statements": [{"medication": "aspirin", "dose": "81mg"}],
    }
    output, prov, auth, schema_ok, errs = task.execute("prov_001", case)
    ok, prov_errs = prov.validate_completeness(manifest)
    results.check("conformant_provenance_complete", ok, str(prov_errs))

    # Check all required fields are present
    d = prov.to_dict()
    results.check("prov_has_case_id", d["case_id"] == "prov_001")
    results.check("prov_has_workflow_id", d["workflow_id"] == "med-recon-poc-v1")
    results.check("prov_has_workflow_version", d["workflow_version"] == "1.0.0")
    results.check("prov_has_fixture_id", d["fixture_id"] == "conformant_fixture")
    results.check("prov_has_model_id", d["model_id"] == "deterministic-conformant-v1")
    results.check("prov_has_adapter_version", d["adapter_version"] == "1.0.0")
    results.check("prov_has_timestamp", d["timestamp"] is not None and len(d["timestamp"]) > 0)
    results.check("prov_has_input_hash", d["input_hash"] is not None and len(d["input_hash"]) == 64)
    results.check("prov_has_raw_output_hash", d["raw_output_hash"] is not None and len(d["raw_output_hash"]) == 64)
    results.check("prov_has_output_hash", d["output_hash"] is not None and len(d["output_hash"]) == 64)
    results.check("prov_has_schema_validation_outcome", d["schema_validation_outcome"] == "PASS")
    results.check("prov_has_auth_outcome", d["authorization_outcome"] == "permitted")
    results.check("prov_has_execution_status", d["execution_status"] == "COMPLETED")

    # Provenance-failure fixture: should fail validation
    adapter2 = ProvenanceFailureFixture()
    task2 = MedicationReconciliationTask(manifest, adapter2)
    output2, prov2, auth2, schema_ok2, errs2 = task2.execute("prov_002", case)
    ok2, prov_errs2 = prov2.validate_completeness(manifest)
    results.check("provenance_failure_detected", not ok2,
                  f"ok={ok2}, errs={prov_errs2}")


def test_synthetic_data_generation(results):
    print("\n--- Synthetic Data Tests ---")
    cases = generate_evaluation_cases()
    results.check("eval_case_count", len(cases) == 60,
                  f"got {len(cases)}")

    auth_cases = [c for c in cases if c["authorization_test"]]
    results.check("auth_cases_exist", len(auth_cases) == 5,
                  f"got {len(auth_cases)}")

    disc_cases = [c for c in cases if c["has_discrepancy"]]
    results.check("discrepancy_cases_exist", len(disc_cases) > 0,
                  f"got {len(disc_cases)}")

    no_disc_cases = [c for c in cases
                     if not c["has_discrepancy"] and not c["authorization_test"]]
    results.check("no_discrepancy_cases_exist", len(no_disc_cases) > 0,
                  f"got {len(no_disc_cases)}")

    # All non-auth cases have ground truth
    for c in cases:
        if not c["authorization_test"]:
            results.check(f"ground_truth_{c['case_id']}",
                          c["ground_truth"] is not None)


def test_hash_determinism(results):
    print("\n--- Hash Determinism Tests ---")
    data = {"a": 1, "b": [2, 3]}
    h1 = hash_data(data)
    h2 = hash_data(data)
    results.check("hash_deterministic", h1 == h2)

    data2 = {"b": [2, 3], "a": 1}  # Different key order
    h3 = hash_data(data2)
    results.check("hash_key_order_invariant", h1 == h3)


def test_authorization_negative_cases(results):
    print("\n--- Authorization Negative Test Cases ---")
    manifest = load_manifest()
    cases = generate_evaluation_cases()
    auth_cases = [c for c in cases if c["authorization_test"]]

    for fixture_cls in [ConformantFixture, SchemaFailureFixture,
                        PerformanceFailureFixture, ProvenanceFailureFixture]:
        adapter = fixture_cls()
        task = MedicationReconciliationTask(manifest, adapter)

        for case in auth_cases:
            output, prov, auth, schema_ok, errs = task.execute(
                case["case_id"], case["input"]
            )
            results.check(
                f"auth_blocked_{adapter.FIXTURE_ID}_{case['case_id']}",
                auth.attempted and auth.blocked and not auth.executed and output is None,
                f"attempted={auth.attempted}, blocked={auth.blocked}, "
                f"executed={auth.executed}, output={output}"
            )


def test_harness_negative_logic(results):
    print("\n--- Harness Negative Logic Tests ---")
    manifest = load_manifest()
    
    # SPST-4 Negative: If provenance_failure is complete, it should FAIL
    # We can fake it by passing only ConformantFixture
    res4 = run_spst.run_spst4(manifest, {"conformant": ConformantFixture()})
    results.check("spst4_fails_when_invariant_violated", res4["pass"] == False, "SPST-4 should fail if provenance_failure fixture is missing or doesn't fail")

    # SPST-5 Negative: If schema_failure passes thresholds, it should FAIL
    # We fake this by making conformant fixture FAIL (which violates the conformant_passes invariant)
    res5 = run_spst.run_spst5(manifest, {"schema_failure": SchemaFailureFixture()})
    results.check("spst5_fails_when_invariant_violated", res5["pass"] == False, "SPST-5 should fail if conformant fixture is missing or doesn't pass")

    # SPST-5 Negative: If conformant fixture lacks discrepancy precision, it should FAIL
    class PrecisionFailureFixture(ConformantFixture):
        FIXTURE_ID = "precision_failure_fixture"
        def _execute_reconciliation(self, case_id, input_data):
            res = super()._execute_reconciliation(case_id, input_data)
            if res and "discrepancies" in res:
                # Fabricate extra discrepancy to destroy precision
                res["discrepancies"].append("Fabricated discrepancy to drop precision")
            return res

    res5_precision = run_spst.run_spst5(manifest, {"conformant_fixture": PrecisionFailureFixture()})
    results.check("spst5_fails_when_low_precision", res5_precision["pass"] == False, "SPST-5 should fail if precision is below threshold")

    # SPST-5 Negative: If conformant fixture lacks human-review sensitivity, it should FAIL
    class SensitivityFailureFixture(ConformantFixture):
        FIXTURE_ID = "sensitivity_failure_fixture"
        def _execute_reconciliation(self, case_id, input_data):
            res = super()._execute_reconciliation(case_id, input_data)
            if res:
                # Force human review to false to destroy sensitivity
                res["human_review_required"] = False
            return res

    res5_sensitivity = run_spst.run_spst5(manifest, {"conformant_fixture": SensitivityFailureFixture()})
    results.check("spst5_fails_when_low_human_review_sensitivity", res5_sensitivity["pass"] == False, "SPST-5 should fail if human review sensitivity is below threshold")

    # SPST-6 logic is tested via execution. If we change manifest rollback, it fails.
    old_val = manifest["rollback"]["validated_configuration"]
    manifest["rollback"]["validated_configuration"] = "wrong_fixture"
    res6 = run_spst.run_spst6(manifest)
    results.check("spst6_fails_when_invariant_violated", res6["pass"] == False, "SPST-6 should fail if rollback config doesn't match")
    manifest["rollback"]["validated_configuration"] = old_val


if __name__ == "__main__":
    print("SPST Reference Implementation -- Automated Conformance Checks")
    print("=" * 60)

    results = TestResults()

    test_schema_validation(results)
    test_authorization(results)
    test_reconciliation_logic(results)
    test_conformant_fixture(results)
    test_schema_failure_fixture(results)
    test_performance_failure_fixture(results)
    test_provenance_completeness(results)
    test_synthetic_data_generation(results)
    test_hash_determinism(results)
    test_authorization_negative_cases(results)
    test_harness_negative_logic(results)

    all_passed = results.summary()
    sys.exit(0 if all_passed else 1)
