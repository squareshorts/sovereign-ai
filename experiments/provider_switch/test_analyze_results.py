import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.provider_switch.analyze_results import calculate_metrics, bootstrap_ci, PRODUCTION_BOOTSTRAP_RESAMPLES

def generate_synthetic_results(replicates=3, cases_per_rep=210, mode="pass"):
    results = []
    # 210 behavioral cases
    for rep in range(1, replicates + 1):
        for i in range(cases_per_rep):
            truth = {
                "has_discrepancy": i < 118,
                "human_review_expected": i < 118,
                "behavioral_evaluation": True,
                "authorization_evaluation": False,
                "request_meds": [{"medication": f"med_{i}", "dose": "10 mg"}],
                "statement_meds": []
            }
            out = {
                "request_meds": [{"medication": f"med_{i}", "dose": "10 mg"}],
                "statement_meds": [],
                "discrepancies": ["disc"] if i < 118 else [],
                "human_review_required": i < 118
            }
            r = {
                "provider_id": "P1",
                "replicate": rep,
                "case_id": f"case_{i}",
                "execution_status": "COMPLETED_SCHEMA_VALID",
                "schema_valid": True,
                "truth": truth,
                "output": out
            }
            if mode == "fail_sens" and i < 120:
                # 80/120 sensitivity = 0.66
                out["discrepancies"] = ["disc"] if i < 80 else []
                out["human_review_required"] = i < 80
            if mode == "schema_failure":
                r["execution_status"] = "COMPLETED_SCHEMA_FAILURE"
                r["schema_valid"] = False
                
            results.append(r)
    return results

def test_passing_results():
    data = generate_synthetic_results(mode="pass")
    metrics = calculate_metrics(data)
    assert metrics["schema_valid_rate"] == 1.0
    assert metrics["discrepancy_sensitivity"] == 1.0
    assert metrics["human_review_sensitivity"] == 1.0
    assert metrics["critical_error_rate"] == 0.0

def test_failing_sens():
    data = generate_synthetic_results(mode="fail_sens")
    metrics = calculate_metrics(data)
    assert metrics["discrepancy_sensitivity"] < 0.90
    assert metrics["human_review_sensitivity"] < 0.90

def test_schema_failure():
    data = generate_synthetic_results(mode="schema_failure")
    metrics = calculate_metrics(data)
    assert metrics["schema_valid_rate"] == 0.0
    assert metrics["discrepancy_sensitivity"] == 0.0 # treated as FN
    assert metrics["human_review_sensitivity"] == 0.0 # treated as FN

if __name__ == "__main__":
    test_passing_results()
    test_failing_sens()
    test_schema_failure()
    assert PRODUCTION_BOOTSTRAP_RESAMPLES == 10000
    print("Tests passed.")
