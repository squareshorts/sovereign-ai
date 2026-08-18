import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.provider_switch.analyze_results import calculate_metrics, compute_extraction_f1_exact

def test_exact_match_f1():
    true_req = [{"medication": "A", "dose": "10"}, {"medication": "B", "dose": "20"}]
    true_stat = [{"medication": "C", "dose": "30"}]
    
    # Exact match
    pred_req = [{"medication": " A ", "dose": "10 "}, {"medication": "b", "dose": "20"}]
    pred_stat = [{"medication": "C", "dose": "30"}]
    
    tp, fp, fn = compute_extraction_f1_exact(pred_req, pred_stat, true_req, true_stat)
    assert tp == 3
    assert fp == 0
    assert fn == 0
    
    # FP and FN
    pred_req = [{"medication": "A", "dose": "10"}, {"medication": "D", "dose": "40"}]
    pred_stat = []
    
    tp, fp, fn = compute_extraction_f1_exact(pred_req, pred_stat, true_req, true_stat)
    assert tp == 1 # A
    assert fp == 1 # D
    assert fn == 2 # B, C

def test_failed_case_itt():
    sample = [{
        "case_id": "c1",
        "schema_valid": False,
        "execution_status": "FAILED_PARSE",
        "extracted_output": None,
        "truth": {
            "has_discrepancy": True,
            "human_review_expected": False,
            "request_meds": [{"medication": "A", "dose": "10"}],
            "statement_meds": []
        }
    }]
    metrics = calculate_metrics(sample)
    
    assert metrics["discrepancy_sensitivity"] == 0.0 # 0 / (0 + 1)
    assert metrics["human_review_sensitivity"] is None # TN logic means no FN because not expected
    # wait, if has_hr_truth is False, hr_tn = 1. So hr_tp = 0, hr_fn = 0. hr_sens = None.
    
    assert metrics["extraction_f1"] == 0.0
    
def test_schema_valid_extraction():
    sample = [{
        "case_id": "c1",
        "schema_valid": True,
        "execution_status": "COMPLETED_SCHEMA_VALID",
        "extracted_output": {
            "request_meds": [{"medication": "A", "dose": "10"}],
            "statement_meds": []
        },
        "workflow_output": {
            "discrepancies": ["disc1"],
            "human_review_required": True
        },
        "truth": {
            "has_discrepancy": True,
            "human_review_expected": True,
            "request_meds": [{"medication": "A", "dose": "10"}],
            "statement_meds": []
        }
    }]
    metrics = calculate_metrics(sample)
    
    assert metrics["discrepancy_sensitivity"] == 1.0
    assert metrics["human_review_sensitivity"] == 1.0
    assert metrics["extraction_f1"] == 1.0

if __name__ == "__main__":
    test_exact_match_f1()
    test_failed_case_itt()
    test_schema_valid_extraction()
    print("All analyzer tests passed")
