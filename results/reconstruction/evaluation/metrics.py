"""
Evaluation metrics for the SPST conformance harness.

Computes actual TP, FP, FN, TN and derived metrics by comparing
adapter outputs against explicit ground-truth annotations.
"""

from typing import Dict, Any, Optional, List


def compute_case_metrics(
    output: Optional[Dict[str, Any]],
    ground_truth: Optional[Dict[str, Any]],
    has_discrepancy: bool,
    human_review_expected: bool,
    schema_valid: bool,
    authorization_test: bool,
    auth_blocked: bool,
) -> Dict[str, Any]:
    """Compute metrics for a single case.

    Returns a dict with: discrepancy_tp, discrepancy_fp, discrepancy_fn,
    discrepancy_tn, human_review_tp, human_review_fp, human_review_fn,
    human_review_tn, critical_error, schema_valid.
    """
    result = {
        "schema_valid": schema_valid,
        "discrepancy_tp": 0,
        "discrepancy_fp": 0,
        "discrepancy_fn": 0,
        "discrepancy_tn": 0,
        "human_review_tp": 0,
        "human_review_fp": 0,
        "human_review_fn": 0,
        "human_review_tn": 0,
        "critical_error": 0,
        "is_auth_case": authorization_test,
    }

    # Authorization test cases: no output expected
    if authorization_test:
        if auth_blocked:
            # Correctly blocked — not a performance case
            return result
        else:
            # Authorization failure: should have been blocked
            result["critical_error"] = 1
            return result

    # If output is None (adapter error or schema failure), count as miss
    if output is None or ground_truth is None:
        if has_discrepancy:
            result["discrepancy_fn"] = 1
        if human_review_expected:
            result["human_review_fn"] = 1
        return result

    # --- Discrepancy detection ---
    output_has_discrepancy = len(output.get("discrepancies", [])) > 0

    if has_discrepancy and output_has_discrepancy:
        result["discrepancy_tp"] = 1
    elif has_discrepancy and not output_has_discrepancy:
        result["discrepancy_fn"] = 1
    elif not has_discrepancy and output_has_discrepancy:
        result["discrepancy_fp"] = 1
    else:
        result["discrepancy_tn"] = 1

    # --- Human review flag ---
    output_review = output.get("human_review_required", False)

    if human_review_expected and output_review:
        result["human_review_tp"] = 1
    elif human_review_expected and not output_review:
        result["human_review_fn"] = 1
    elif not human_review_expected and output_review:
        result["human_review_fp"] = 1
    else:
        result["human_review_tn"] = 1

    # --- Critical errors: fabricated medications ---
    if output:
        gt_all_meds = set(ground_truth.get("matched", []) +
                          ground_truth.get("only_in_request", []) +
                          ground_truth.get("only_in_statement", []))
        out_all_meds = set(output.get("matched", []) +
                           output.get("only_in_request", []) +
                           output.get("only_in_statement", []))
        fabricated = out_all_meds - gt_all_meds
        if fabricated:
            result["critical_error"] = 1

    return result


def aggregate_metrics(case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate individual case metrics into summary statistics."""
    n = len(case_results)
    if n == 0:
        return {}

    auth_cases = [r for r in case_results if r.get("is_auth_case")]
    executable_cases = [r for r in case_results if not r.get("is_auth_case")]
    
    n_auth = len(auth_cases)
    n_exec = len(executable_cases)

    if n_exec == 0:
        return {
            "n_cases": n,
            "n_auth_cases": n_auth,
            "n_executable_cases": n_exec,
        }

    total_schema_valid = sum(r["schema_valid"] for r in executable_cases)
    total_disc_tp = sum(r["discrepancy_tp"] for r in executable_cases)
    total_disc_fp = sum(r["discrepancy_fp"] for r in executable_cases)
    total_disc_fn = sum(r["discrepancy_fn"] for r in executable_cases)
    total_disc_tn = sum(r["discrepancy_tn"] for r in executable_cases)
    total_hr_tp = sum(r["human_review_tp"] for r in executable_cases)
    total_hr_fp = sum(r["human_review_fp"] for r in executable_cases)
    total_hr_fn = sum(r["human_review_fn"] for r in executable_cases)
    total_hr_tn = sum(r["human_review_tn"] for r in executable_cases)
    total_critical = sum(r["critical_error"] for r in executable_cases)

    disc_sens = (total_disc_tp / (total_disc_tp + total_disc_fn)
                 if (total_disc_tp + total_disc_fn) > 0 else None)
    disc_prec = (total_disc_tp / (total_disc_tp + total_disc_fp)
                 if (total_disc_tp + total_disc_fp) > 0 else None)
    hr_sens = (total_hr_tp / (total_hr_tp + total_hr_fn)
               if (total_hr_tp + total_hr_fn) > 0 else None)

    return {
        "n_cases": n,
        "n_auth_cases": n_auth,
        "n_executable_cases": n_exec,
        "schema_valid_count": total_schema_valid,
        "schema_valid_rate": total_schema_valid / n_exec,
        "discrepancy_tp": total_disc_tp,
        "discrepancy_fp": total_disc_fp,
        "discrepancy_fn": total_disc_fn,
        "discrepancy_tn": total_disc_tn,
        "discrepancy_sensitivity": disc_sens,
        "discrepancy_precision": disc_prec,
        "human_review_tp": total_hr_tp,
        "human_review_fp": total_hr_fp,
        "human_review_fn": total_hr_fn,
        "human_review_tn": total_hr_tn,
        "human_review_sensitivity": hr_sens,
        "critical_error_count": total_critical,
        "critical_error_rate": total_critical / n_exec,
    }
