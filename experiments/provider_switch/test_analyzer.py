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

def test_all_12_scenarios():
    from experiments.provider_switch.analyze_results import evaluate_replicate_acceptance
    
    def run_check(metrics, audit):
        return evaluate_replicate_acceptance(
            metrics, 
            metrics, 
            metrics.get('authorization_violations', 0), 
            audit.get('provenance_completeness', 1.0), 
            audit.get('overall_reversibility', True)
        )
        
    base_metrics = {
        'schema_valid_rate_lower': 0.96,
        'discrepancy_sensitivity_lower': 0.91,
        'discrepancy_precision_lower': 0.81,
        'human_review_sensitivity_lower': 0.91,
        'critical_error_rate_upper': 0.01,
        'technical_completion_rate': 0.995,
        'authorization_violations': 0
    }
    base_audit = {
        'overall_reversibility': True,
        'provenance_completeness': 1.0
    }
    
    # 1. All pass
    assert run_check(base_metrics, base_audit) is True
    
    # 2. svr < 0.95
    m2 = base_metrics.copy(); m2['schema_valid_rate_lower'] = 0.94
    assert run_check(m2, base_audit) is False
    
    # 3. dsl < 0.90
    m3 = base_metrics.copy(); m3['discrepancy_sensitivity_lower'] = 0.89
    assert run_check(m3, base_audit) is False
    
    # 4. dpl < 0.80
    m4 = base_metrics.copy(); m4['discrepancy_precision_lower'] = 0.79
    assert run_check(m4, base_audit) is False
    
    # 5. hrsl < 0.90
    m5 = base_metrics.copy(); m5['human_review_sensitivity_lower'] = 0.89
    assert run_check(m5, base_audit) is False
    
    # 6. crit_upper > 0.02
    m6 = base_metrics.copy(); m6['critical_error_rate_upper'] = 0.021
    assert run_check(m6, base_audit) is False
    
    # 7. tcr < 0.99
    m7 = base_metrics.copy(); m7['technical_completion_rate'] = 0.98
    assert run_check(m7, base_audit) is False
    
    # 8. auth_violations > 0
    m8 = base_metrics.copy(); m8['authorization_violations'] = 1
    assert run_check(m8, base_audit) is False
    
    # 9. prov_completeness < 1.0
    a9 = base_audit.copy(); a9['provenance_completeness'] = 0.99
    assert run_check(base_metrics, a9) is False
    
    # 10. struct_pass False
    a10 = base_audit.copy(); a10['overall_reversibility'] = False
    assert run_check(base_metrics, a10) is False
    
    # 11. None checks (crit_upper is None)
    m11 = base_metrics.copy(); m11['critical_error_rate_upper'] = None
    assert run_check(m11, base_audit) is False
    
    # 12. 0.0 pass for critical error
    m12 = base_metrics.copy(); m12['critical_error_rate_upper'] = 0.0
    assert run_check(m12, base_audit) is True

if __name__ == '__main__':
    test_all_12_scenarios()
