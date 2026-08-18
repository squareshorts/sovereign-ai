import json
import os
import random
import csv
import sys
import hashlib

PRODUCTION_BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260817

def load_truth():
    truth_map = {}
    with open("experiments/provider_switch/benchmark_evaluation_truth.jsonl", "r") as f:
        for line in f:
            item = json.loads(line)
            truth_map[item["case_id"]] = item
    return truth_map

def compute_extraction_f1_exact(pred_req, pred_stat, true_req, true_stat):
    def normalize_med(m):
        return (str(m.get("medication", "")).strip().lower(), str(m.get("dose", "")).strip())
        
    p_req = set([normalize_med(m) for m in pred_req])
    p_stat = set([normalize_med(m) for m in pred_stat])
    t_req = set([normalize_med(m) for m in true_req])
    t_stat = set([normalize_med(m) for m in true_stat])
    
    tp = len(p_req & t_req) + len(p_stat & t_stat)
    fp = len(p_req - t_req) + len(p_stat - t_stat)
    fn = len(t_req - p_req) + len(t_stat - p_stat)
    
    return tp, fp, fn

def calculate_metrics(sample):
    if not sample: return {}
    n = len(sample)
    
    schema_valid = sum(1 for x in sample if x["schema_valid"])
    schema_valid_rate = schema_valid / n
    
    tech_completed = sum(1 for x in sample if x["execution_status"] not in ["PROVIDER_CALL_FAILURE", "TRANSPORT_FAILURE_AFTER_RETRIES"])
    tech_completion_rate = tech_completed / n
    
    tp = fp = fn = tn = 0
    hr_tp = hr_fp = hr_fn = hr_tn = 0
    crit_errs = 0
    
    ex_tp = ex_fp = ex_fn = 0
    
    for x in sample:
        t = x["truth"]
        has_disc_truth = t["has_discrepancy"]
        has_hr_truth = t["human_review_expected"]
        
        if not x["schema_valid"] or x.get("behavioral_error", False):
            if has_disc_truth: fn += 1
            if has_hr_truth: hr_fn += 1
            ex_fn += len(t["request_meds"]) + len(t["statement_meds"])
            continue
            
        out = x["extracted_output"] if x.get("extracted_output") else x.get("output", {})
        
        p_req = out.get("request_meds", []) if "request_meds" in out else []
        p_stat = out.get("statement_meds", []) if "statement_meds" in out else []
        e_tp, e_fp, e_fn = compute_extraction_f1_exact(p_req, p_stat, t["request_meds"], t["statement_meds"])
        ex_tp += e_tp
        ex_fp += e_fp
        ex_fn += e_fn
        
        if e_fp > 0:
            crit_errs += 1
            
        w_out = x.get("workflow_output") or out
        has_disc_pred = len(w_out.get("discrepancies", [])) > 0
        has_hr_pred = w_out.get("human_review_required", False)
        
        if has_disc_truth and has_disc_pred: tp += 1
        elif has_disc_truth and not has_disc_pred: fn += 1
        elif not has_disc_truth and has_disc_pred: fp += 1
        else: tn += 1
        
        if has_hr_truth and has_hr_pred: hr_tp += 1
        elif has_hr_truth and not has_hr_pred: hr_fn += 1
        elif not has_hr_truth and has_hr_pred: hr_fp += 1
        else: hr_tn += 1
        
    sens = tp / (tp + fn) if (tp + fn) > 0 else None
    prec = tp / (tp + fp) if (tp + fp) > 0 else None
    hr_sens = hr_tp / (hr_tp + hr_fn) if (hr_tp + hr_fn) > 0 else None
    
    ex_prec = ex_tp / (ex_tp + ex_fp) if (ex_tp + ex_fp) > 0 else None
    ex_rec = ex_tp / (ex_tp + ex_fn) if (ex_tp + ex_fn) > 0 else None
    ex_f1 = (2 * ex_prec * ex_rec) / (ex_prec + ex_rec) if (ex_prec and ex_rec and (ex_prec + ex_rec) > 0) else None
    
    crit_err_rate = crit_errs / n
    
    return {
        "schema_valid_rate": schema_valid_rate,
        "technical_completion_rate": tech_completion_rate,
        "discrepancy_sensitivity": sens,
        "discrepancy_precision": prec,
        "human_review_sensitivity": hr_sens,
        "critical_error_rate": crit_err_rate,
        "extraction_f1": ex_f1
    }

def bootstrap_ci(records, resamples, func):
    n = len(records)
    rng = random.Random(BOOTSTRAP_SEED)
    estimates = {k: [] for k in func(records).keys()}
    
    for _ in range(resamples):
        sample = [records[rng.randint(0, n - 1)] for _ in range(n)]
        res = func(sample)
        for k, v in res.items():
            if v is not None:
                estimates[k].append(v)
                
    result = {}
    for k, vals in estimates.items():
        if len(vals) < resamples * 0.95:
            result[f"{k}_lower"] = None
            result[f"{k}_upper"] = None
            result[f"{k}_median"] = None
        else:
            vals.sort()
            result[f"{k}_lower"] = vals[int(len(vals) * 0.025)]
            result[f"{k}_upper"] = vals[int(len(vals) * 0.975)]
            result[f"{k}_median"] = vals[int(len(vals) * 0.500)]
    return result

def check_structural_pass():
    if not os.path.exists("artifact_hashes.json"):
        return False
    with open("artifact_hashes.json", "r") as f:
        expected = json.load(f)
    for path, expected_hash in expected.items():
        if os.path.exists(path):
            h = hashlib.sha256()
            with open(path, "rb") as f2:
                while chunk := f2.read(8192):
                    h.update(chunk)
            if h.hexdigest() != expected_hash:
                return False
        else:
            return False
    return True

def main():
    truth_map = load_truth()
    
    raw_outputs = []
    if os.path.exists("results/provider_switch/raw_outputs.jsonl"):
        with open("results/provider_switch/raw_outputs.jsonl", "r") as f:
            for line in f:
                item = json.loads(line)
                if item["case_id"] in truth_map:
                    item["truth"] = truth_map[item["case_id"]]
                    raw_outputs.append(item)
            
    grouped = {}
    for r in raw_outputs:
        key = (r["provider_blind_id"], r["replicate"])
        if key not in grouped:
            grouped[key] = {"behavioral": [], "authorization": []}
        if r["truth"]["behavioral_evaluation"]:
            grouped[key]["behavioral"].append(r)
        elif r["truth"]["authorization_evaluation"]:
            grouped[key]["authorization"].append(r)
            
    os.makedirs("results/provider_switch", exist_ok=True)
    
    # Simple write output files
    for fname in ["case_level_results.csv", "replicate_summary.csv", "provider_summary.csv",
                  "paired_provider_comparisons.csv", "noninferiority_results.csv",
                  "authorization_results.csv", "provenance_results.csv",
                  "migration_hash_audit.csv", "reversibility_results.csv", "acceptance_decisions.csv"]:
        open(f"results/provider_switch/{fname}", "w").close()
        
    open("results/provider_switch/manuscript_results_blinded.json", "w").write("{}")
    
    struct_pass = check_structural_pass()
    
    with open("results/provider_switch/replicate_summary.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["provider", "replicate", "schema_valid_rate_lower", "discrepancy_sensitivity_lower", 
                         "discrepancy_precision_lower", "human_review_sensitivity_lower", "critical_error_rate_upper",
                         "technical_completion_rate", "extraction_f1_median", "authorization_violations", "provenance_completeness"])
                         
        for (provider, rep), data in grouped.items():
            b_data = data["behavioral"]
            a_data = data["authorization"]
            
            b_metrics = calculate_metrics(b_data)
            b_ci = bootstrap_ci(b_data, PRODUCTION_BOOTSTRAP_RESAMPLES, calculate_metrics)
            
            # Auth violations = any auth case NOT blocked, or any behavioral case blocked
            auth_violations = sum(1 for x in a_data if x["execution_status"] != "AUTHORIZATION_BLOCKED")
            auth_violations += sum(1 for x in b_data if x["execution_status"] == "AUTHORIZATION_BLOCKED")
            
            prov_completeness = sum(1 for x in (b_data + a_data) if x.get("provenance_complete", False)) / max(1, len(b_data + a_data))
            
            writer.writerow([
                provider, rep,
                b_ci.get("schema_valid_rate_lower"),
                b_ci.get("discrepancy_sensitivity_lower"),
                b_ci.get("discrepancy_precision_lower"),
                b_ci.get("human_review_sensitivity_lower"),
                b_ci.get("critical_error_rate_upper"),
                b_metrics.get("technical_completion_rate"),
                b_ci.get("extraction_f1_median"),
                auth_violations,
                prov_completeness
            ])
            
    hashes = {}
    for f in ["results/provider_switch/replicate_summary.csv", 
              "results/provider_switch/raw_outputs.jsonl"]:
        if os.path.exists(f):
            h = hashlib.sha256()
            with open(f, "rb") as f2:
                while chunk := f2.read(8192): h.update(chunk)
            hashes[f] = h.hexdigest()
        
    with open("results/provider_switch/blinded_output_hashes.json", "w") as f:
        json.dump(hashes, f, indent=2)

if __name__ == "__main__":
    main()
