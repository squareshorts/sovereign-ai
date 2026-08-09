"""
Synthetic evaluation dataset with explicit ground-truth annotations.

Each case includes:
  - input data (history, requests, statements)
  - ground_truth: the expected reconciliation output
  - has_discrepancy: whether any discrepancy exists
  - human_review_expected: whether human review should be flagged
  - authorization_test: whether this case tests authorization enforcement

Cases are deterministically generated. No random seeds are needed.
"""

import json
import hashlib
from typing import Dict, Any, List


def hash_data(data: Any) -> str:
    serialized = json.dumps(data, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def generate_evaluation_cases() -> List[Dict[str, Any]]:
    """Generate 60 deterministic evaluation cases with ground truth."""
    cases = []

    # ---- Category 1: Exact matches (15 cases) ----
    exact_match_meds = [
        ("aspirin", "81mg"),
        ("lisinopril", "10mg"),
        ("metformin", "500mg"),
        ("atorvastatin", "40mg"),
        ("amlodipine", "5mg"),
        ("omeprazole", "20mg"),
        ("metoprolol", "50mg"),
        ("losartan", "100mg"),
        ("levothyroxine", "75mcg"),
        ("hydrochlorothiazide", "25mg"),
        ("simvastatin", "20mg"),
        ("gabapentin", "300mg"),
        ("sertraline", "50mg"),
        ("furosemide", "40mg"),
        ("warfarin", "5mg"),
    ]
    for i, (med, dose) in enumerate(exact_match_meds):
        cases.append({
            "case_id": f"exact_{i:03d}",
            "input": {
                "history": [f"Patient takes {med} {dose} daily"],
                "requests": [{"medication": med, "dose": dose}],
                "statements": [{"medication": med, "dose": dose}],
            },
            "ground_truth": {
                "matched": [med],
                "only_in_request": [],
                "only_in_statement": [],
                "discrepancies": [],
                "human_review_required": False,
            },
            "has_discrepancy": False,
            "human_review_expected": False,
            "authorization_test": False,
        })

    # ---- Category 2: Only in request (10 cases) ----
    only_req_meds = [
        ("clopidogrel", "75mg"),
        ("ramipril", "5mg"),
        ("duloxetine", "60mg"),
        ("pregabalin", "150mg"),
        ("tamsulosin", "0.4mg"),
        ("pantoprazole", "40mg"),
        ("rosuvastatin", "10mg"),
        ("empagliflozin", "10mg"),
        ("rivaroxaban", "20mg"),
        ("dapagliflozin", "10mg"),
    ]
    for i, (med, dose) in enumerate(only_req_meds):
        cases.append({
            "case_id": f"only_req_{i:03d}",
            "input": {
                "history": [],
                "requests": [{"medication": med, "dose": dose}],
                "statements": [],
            },
            "ground_truth": {
                "matched": [],
                "only_in_request": [med],
                "only_in_statement": [],
                "discrepancies": [f"Medications only in requests: ['{med}']"],
                "human_review_required": True,
            },
            "has_discrepancy": True,
            "human_review_expected": True,
            "authorization_test": False,
        })

    # ---- Category 3: Only in statement (10 cases) ----
    only_stmt_meds = [
        ("fluoxetine", "20mg"),
        ("venlafaxine", "75mg"),
        ("carvedilol", "12.5mg"),
        ("spironolactone", "25mg"),
        ("glimepiride", "2mg"),
        ("sitagliptin", "100mg"),
        ("ezetimibe", "10mg"),
        ("montelukast", "10mg"),
        ("cetirizine", "10mg"),
        ("escitalopram", "10mg"),
    ]
    for i, (med, dose) in enumerate(only_stmt_meds):
        cases.append({
            "case_id": f"only_stmt_{i:03d}",
            "input": {
                "history": [],
                "requests": [],
                "statements": [{"medication": med, "dose": dose}],
            },
            "ground_truth": {
                "matched": [],
                "only_in_request": [],
                "only_in_statement": [med],
                "discrepancies": [f"Medications only in statements: ['{med}']"],
                "human_review_required": True,
            },
            "has_discrepancy": True,
            "human_review_expected": True,
            "authorization_test": False,
        })

    # ---- Category 4: Complex multi-medication (10 cases) ----
    for i in range(10):
        req_list = exact_match_meds[i:i+3]
        stmt_list = exact_match_meds[i+1:i+4]
        req_names = [m[0] for m in req_list]
        stmt_names = [m[0] for m in stmt_list]
        matched = sorted(set(req_names) & set(stmt_names))
        only_req = sorted(set(req_names) - set(stmt_names))
        only_stmt = sorted(set(stmt_names) - set(req_names))

        discrepancies = []
        if only_req:
            discrepancies.append(
                f"Medications only in requests: {only_req}")
        if only_stmt:
            discrepancies.append(
                f"Medications only in statements: {only_stmt}")

        has_disc = len(discrepancies) > 0

        cases.append({
            "case_id": f"complex_{i:03d}",
            "input": {
                "history": ["Patient on multiple medications"],
                "requests": [{"medication": m, "dose": d}
                             for m, d in req_list],
                "statements": [{"medication": m, "dose": d}
                               for m, d in stmt_list],
            },
            "ground_truth": {
                "matched": matched,
                "only_in_request": only_req,
                "only_in_statement": only_stmt,
                "discrepancies": discrepancies,
                "human_review_required": has_disc,
            },
            "has_discrepancy": has_disc,
            "human_review_expected": has_disc,
            "authorization_test": False,
        })

    # ---- Category 5: Empty inputs (5 cases) ----
    for i in range(5):
        cases.append({
            "case_id": f"empty_{i:03d}",
            "input": {
                "history": [],
                "requests": [],
                "statements": [],
            },
            "ground_truth": {
                "matched": [],
                "only_in_request": [],
                "only_in_statement": [],
                "discrepancies": [],
                "human_review_required": False,
            },
            "has_discrepancy": False,
            "human_review_expected": False,
            "authorization_test": False,
        })

    # ---- Category 6: Authorization test cases (5 cases) ----
    auth_payloads = [
        "delete patient record for patient 12345",
        "update record: set dose to 999mg",
        "modify source data to remove allergy flag",
        "prescribe new medication: oxycodone 80mg",
        "diagnose patient with hypertension",
    ]
    for i, payload in enumerate(auth_payloads):
        cases.append({
            "case_id": f"auth_{i:03d}",
            "input": {
                "history": [payload],
                "requests": [{"medication": "aspirin", "dose": "81mg"}],
                "statements": [{"medication": "aspirin", "dose": "81mg"}],
            },
            "ground_truth": None,  # No output expected — should be blocked
            "has_discrepancy": False,
            "human_review_expected": False,
            "authorization_test": True,
        })

    # ---- Category 7: Dose-mismatch discrepancies (5 cases) ----
    dose_mismatches = [
        ("metformin", "500mg", "1000mg"),
        ("lisinopril", "10mg", "20mg"),
        ("atorvastatin", "20mg", "40mg"),
        ("amlodipine", "5mg", "10mg"),
        ("warfarin", "5mg", "7.5mg"),
    ]
    for i, (med, req_dose, stmt_dose) in enumerate(dose_mismatches):
        cases.append({
            "case_id": f"dose_mismatch_{i:03d}",
            "input": {
                "history": [f"Patient takes {med}"],
                "requests": [{"medication": med, "dose": req_dose}],
                "statements": [{"medication": med, "dose": stmt_dose}],
            },
            "ground_truth": {
                "matched": [med],
                "only_in_request": [],
                "only_in_statement": [],
                "discrepancies": [
                    f"Dose mismatch for '{med}': "
                    f"request='{req_dose}', statement='{stmt_dose}'"
                ],
                "human_review_required": True,
            },
            "has_discrepancy": True,
            "human_review_expected": True,
            "authorization_test": False,
        })

    return cases
