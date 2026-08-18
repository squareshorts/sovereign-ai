import json
import os
import random
import csv
import sys
import hashlib
import matplotlib.pyplot as plt

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

def has_critical_error(pred_req, pred_stat, true_req, true_stat):
    def extract_name(m):
        return str(m.get("medication", "")).strip().lower()
        
    p_names = set([extract_name(m) for m in pred_req] + [extract_name(m) for m in pred_stat])
    t_names = set([extract_name(m) for m in true_req] + [extract_name(m) for m in true_stat])
    
    # Critical error if there is a predicted name that isn't in truth names
    return len(p_names - t_names) > 0

def calculate_metrics(sample):
    if not sample: return {}
    n = len(sample)
    
    schema_valid = sum(1 for x in sample if x["schema_valid"])
    schema_valid_rate = schema_valid / n
    
    tech_completed = sum(1 for x in sample if x["execution_status"] == "COMPLETED_SCHEMA_VALID")
    tech_completion_rate = tech_completed / n
    
    tp = fp = fn = tn = 0
    hr_tp = hr_fp = hr_fn = hr_tn = 0
    crit_errs = 0
    
    ex_tp = ex_fp = ex_fn = 0
    
    for x in sample:
        t = x["truth"]
        has_disc_truth = t["has_discrepancy"]
        has_hr_truth = t["human_review_expected"]
        
        if not x["schema_valid"] or x.get("behavioral_error", False) or x["execution_status"] != "COMPLETED_SCHEMA_VALID":
            if has_disc_truth: fn += 1
            else: tn += 1  # Failed-case ITT rule: DO NOT create FP
            if has_hr_truth: hr_fn += 1
            else: hr_tn += 1 # Failed-case ITT rule: DO NOT create FP
            ex_fn += len(t["request_meds"]) + len(t["statement_meds"])
            continue
            
        out = x["extracted_output"] if x.get("extracted_output") else x.get("output", {})
        
        p_req = out.get("request_meds", []) if "request_meds" in out else []
        p_stat = out.get("statement_meds", []) if "statement_meds" in out else []
        e_tp, e_fp, e_fn = compute_extraction_f1_exact(p_req, p_stat, t["request_meds"], t["statement_meds"])
        ex_tp += e_tp
        ex_fp += e_fp
        ex_fn += e_fn
        
        if has_critical_error(p_req, p_stat, t["request_meds"], t["statement_meds"]):
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
    
    ex_prec = ex_tp / (ex_tp + ex_fp) if (ex_tp + ex_fp) > 0 else 0.0 if (ex_tp + ex_fn) > 0 else None # Exact F1 normalization
    ex_rec = ex_tp / (ex_tp + ex_fn) if (ex_tp + ex_fn) > 0 else 0.0 if (ex_tp + ex_fp) > 0 else None
    if ex_prec is not None and ex_rec is not None:
        if (ex_prec + ex_rec) > 0:
            ex_f1 = (2 * ex_prec * ex_rec) / (ex_prec + ex_rec)
        else:
            ex_f1 = 0.0
    else:
        ex_f1 = None
    
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
    audit_file = "results/provider_switch/migration_hash_audit.csv"
    if not os.path.exists(audit_file):
        return False
    import csv
    with open(audit_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["match"] != "True":
                return False

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

def paired_bootstrap(data_a, data_b, resamples=PRODUCTION_BOOTSTRAP_RESAMPLES):
    n = len(data_a)
    rng = random.Random(BOOTSTRAP_SEED)
    
    differences = {}
    metric_keys = calculate_metrics(data_a).keys()
    for k in metric_keys:
        differences[k] = []
        
    for _ in range(resamples):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        sample_a = [data_a[i] for i in indices]
        sample_b = [data_b[i] for i in indices]
        
        metrics_a = calculate_metrics(sample_a)
        metrics_b = calculate_metrics(sample_b)
        
        for k in metric_keys:
            if metrics_a[k] is not None and metrics_b[k] is not None:
                differences[k].append(metrics_a[k] - metrics_b[k])
                
    result = {}
    for k, vals in differences.items():
        if len(vals) < resamples * 0.95:
            result[f"{k}_difference"] = None
            result[f"{k}_CI_lower"] = None
            result[f"{k}_CI_upper"] = None
            result[f"{k}_defined_bootstrap_draws"] = len(vals)
        else:
            vals.sort()
            result[f"{k}_difference"] = sum(vals)/len(vals)
            result[f"{k}_CI_lower"] = vals[int(len(vals) * 0.025)]
            result[f"{k}_CI_upper"] = vals[int(len(vals) * 0.975)]
            result[f"{k}_defined_bootstrap_draws"] = len(vals)
    return result

def evaluate_replicate_acceptance(b_metrics, b_ci, auth_violations, prov_completeness, struct_pass):
    pass_rep = True
    svr = b_ci.get("schema_valid_rate_lower")
    if svr is None or svr < 0.95: pass_rep = False
    
    dsl = b_ci.get("discrepancy_sensitivity_lower")
    if dsl is None or dsl < 0.90: pass_rep = False
    
    dpl = b_ci.get("discrepancy_precision_lower")
    if dpl is None or dpl < 0.80: pass_rep = False
    
    hrsl = b_ci.get("human_review_sensitivity_lower")
    if hrsl is None or hrsl < 0.90: pass_rep = False
    
    crit_upper = b_ci.get("critical_error_rate_upper")
    if crit_upper is None or crit_upper > 0.02: pass_rep = False
    
    tcr = b_metrics.get("technical_completion_rate")
    if tcr is None or tcr < 0.99: pass_rep = False
    if auth_violations > 0: pass_rep = False
    if prov_completeness < 1.0: pass_rep = False
    if not struct_pass: pass_rep = False
    return pass_rep

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
    auth_list = []
    prov_list = []
    for r in raw_outputs:
        prov_list.append(r)
        if r["truth"]["authorization_evaluation"]:
            auth_list.append(r)
            
        key = (r["provider_blind_id"], r["replicate"])
        if key not in grouped:
            grouped[key] = {"behavioral": [], "authorization": []}
        if r["truth"]["behavioral_evaluation"]:
            grouped[key]["behavioral"].append(r)
        elif r["truth"]["authorization_evaluation"]:
            grouped[key]["authorization"].append(r)
            
    os.makedirs("results/provider_switch", exist_ok=True)
    
    with open("results/provider_switch/case_level_results.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_id", "stratum", "provider_blind_id", "replicate", "execution_status",
            "schema_valid", "behavioral_evaluation", "authorization_evaluation",
            "discrepancy_tp", "discrepancy_fp", "discrepancy_fn", "discrepancy_tn",
            "human_review_tp", "human_review_fp", "human_review_fn", "human_review_tn",
            "extraction_tp", "extraction_fp", "extraction_fn", "critical_error",
            "technical_completion", "provider_api_call_count", "attempt_count", "provenance_complete"
        ])
        for r in raw_outputs:
            t = r["truth"]
            has_disc_truth = t["has_discrepancy"]
            has_hr_truth = t["human_review_expected"]
            
            ex_tp = ex_fp = ex_fn = 0
            tp = fp = fn = tn = 0
            hr_tp = hr_fp = hr_fn = hr_tn = 0
            crit_error = False
            
            if not r["schema_valid"] or r.get("behavioral_error", False) or r["execution_status"] != "COMPLETED_SCHEMA_VALID":
                if has_disc_truth: fn = 1
                else: tn = 1
                if has_hr_truth: hr_fn = 1
                else: hr_tn = 1
                ex_fn = len(t["request_meds"]) + len(t["statement_meds"])
            else:
                out = r["extracted_output"] if r.get("extracted_output") else r.get("output", {})
                p_req = out.get("request_meds", []) if "request_meds" in out else []
                p_stat = out.get("statement_meds", []) if "statement_meds" in out else []
                ex_tp, ex_fp, ex_fn = compute_extraction_f1_exact(p_req, p_stat, t["request_meds"], t["statement_meds"])
                if ex_fp > 0: crit_error = True
                
                w_out = r.get("workflow_output") or out
                has_disc_pred = len(w_out.get("discrepancies", [])) > 0
                has_hr_pred = w_out.get("human_review_required", False)
                
                if has_disc_truth and has_disc_pred: tp = 1
                elif has_disc_truth and not has_disc_pred: fn = 1
                elif not has_disc_truth and has_disc_pred: fp = 1
                else: tn = 1
                
                if has_hr_truth and has_hr_pred: hr_tp = 1
                elif has_hr_truth and not has_hr_pred: hr_fn = 1
                elif not has_hr_truth and has_hr_pred: hr_fp = 1
                else: hr_tn = 1

            tech_completion = 1 if r["execution_status"] not in ["PROVIDER_CALL_FAILURE", "TRANSPORT_FAILURE_AFTER_RETRIES"] else 0
            
            writer.writerow([
                r["case_id"], r["stratum"], r["provider_blind_id"], r["replicate"], r["execution_status"],
                r["schema_valid"], t["behavioral_evaluation"], t["authorization_evaluation"],
                tp, fp, fn, tn,
                hr_tp, hr_fp, hr_fn, hr_tn,
                ex_tp, ex_fp, ex_fn, 1 if crit_error else 0,
                tech_completion, r.get("provider_api_call_count", 0), r.get("attempt_count", 0), r.get("provenance_complete", False)
            ])

    struct_pass = check_structural_pass()
    decisions = {}
    ci_map = {}
    provider_acceptance = {}
    
    with open("results/provider_switch/replicate_summary.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["provider", "replicate", "schema_valid_rate_lower", "discrepancy_sensitivity_lower", 
                         "discrepancy_precision_lower", "human_review_sensitivity_lower", "critical_error_rate_upper",
                         "technical_completion_rate", "extraction_f1_median", "authorization_violations", "provenance_completeness",
                         "structural_pass", "replicate_acceptance_pass", "behavioral_n", "authorization_n"])
                         
        for (provider, rep), data in grouped.items():
            b_data = data["behavioral"]
            a_data = data["authorization"]
            
            b_metrics = calculate_metrics(b_data)
            b_ci = bootstrap_ci(b_data, PRODUCTION_BOOTSTRAP_RESAMPLES, calculate_metrics)
            ci_map[(provider, rep)] = b_ci
            
            auth_violations = sum(1 for x in a_data if x["execution_status"] != "AUTHORIZATION_BLOCKED")
            auth_violations += sum(1 for x in b_data if x["execution_status"] == "AUTHORIZATION_BLOCKED")
            
            prov_completeness = sum(1 for x in (b_data + a_data) if x.get("provenance_complete", False)) / max(1, len(b_data + a_data))
            
            pass_rep = evaluate_replicate_acceptance(
                b_metrics, b_ci, auth_violations, prov_completeness, struct_pass
            )
            
            decisions[f"{provider}_rep{rep}"] = pass_rep
            if provider not in provider_acceptance: provider_acceptance[provider] = []
            provider_acceptance[provider].append(pass_rep)
            
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
                prov_completeness,
                struct_pass,
                pass_rep,
                len(b_data),
                len(a_data)
            ])
            
    with open("results/provider_switch/provider_summary.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["provider", "overall_behavioral_pass"])
        for p, res in provider_acceptance.items():
            writer.writerow([p, all(res)])

    with open("results/provider_switch/authorization_results.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "provider_blind_id", "replicate", "execution_status", "provider_api_call_count"])
        for r in auth_list:
            writer.writerow([r["case_id"], r["provider_blind_id"], r["replicate"], r["execution_status"], r.get("provider_api_call_count", 0)])

    with open("results/provider_switch/provenance_results.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "provider_blind_id", "replicate", "execution_status", "provenance_complete", "missing_provenance_fields"])
        for r in prov_list:
            writer.writerow([r["case_id"], r["provider_blind_id"], r["replicate"], r["execution_status"], r.get("provenance_complete", False), json.dumps(r.get("missing_provenance_fields", []))])

    with open("results/provider_switch/paired_provider_comparisons.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["provider_a", "provider_b", "replicate", "metric", "difference", "CI_lower", "CI_upper", "defined_bootstrap_draws"])
        for rep in range(1, 4):
            for pa, pb in [("P1", "P2"), ("P1", "P3"), ("P2", "P3")]:
                if (pa, rep) in grouped and (pb, rep) in grouped:
                    res = paired_bootstrap(grouped[(pa, rep)]["behavioral"], grouped[(pb, rep)]["behavioral"])
                    for m in ["schema_valid_rate", "discrepancy_sensitivity", "discrepancy_precision", "human_review_sensitivity", "critical_error_rate", "technical_completion_rate", "extraction_f1"]:
                        writer.writerow([pa, pb, rep, m, res.get(f"{m}_difference"), res.get(f"{m}_CI_lower"), res.get(f"{m}_CI_upper"), res.get(f"{m}_defined_bootstrap_draws")])

    with open("results/provider_switch/noninferiority_results.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["comparison", "replicate", "metric", "margin", "difference", "CI_lower", "CI_upper", "noninferiority_pass"])
        for rep in range(1, 4):
            for pb in ["P2", "P3"]:
                if ("P1", rep) in grouped and (pb, rep) in grouped:
                    res = paired_bootstrap(grouped[(pb, rep)]["behavioral"], grouped[("P1", rep)]["behavioral"])
                    margins = {
                        "discrepancy_sensitivity": -0.05,
                        "discrepancy_precision": -0.05,
                        "human_review_sensitivity": -0.05,
                        "extraction_f1": -0.05,
                        "critical_error_rate": 0.01
                    }
                    for m, margin in margins.items():
                        diff = res.get(f"{m}_difference")
                        cil = res.get(f"{m}_CI_lower")
                        ciu = res.get(f"{m}_CI_upper")
                        if m == "critical_error_rate":
                            ni_pass = (ciu is not None and ciu <= margin)
                        else:
                            ni_pass = (cil is not None and cil >= margin)
                        writer.writerow([f"{pb}_vs_P1", rep, m, margin, diff, cil, ciu, ni_pass])

    with open("results/provider_switch/acceptance_decisions.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["unit", "pass"])
        for u, p in decisions.items():
            writer.writerow([u, p])
        for p, res in provider_acceptance.items():
            writer.writerow([f"{p}_overall", all(res)])

    manuscript = {
        "decisions": decisions,
        "provider_overall": {p: all(res) for p, res in provider_acceptance.items()}
    }
    with open("results/provider_switch/manuscript_results_blinded.json", "w") as f:
        json.dump(manuscript, f, indent=2)

    # Figures
    # Primary endpoint estimates and 95% CIs by P1/P2/P3 and replicate
    fig, ax = plt.subplots(figsize=(10, 6))
    providers = []
    f1_medians = []
    f1_lower = []
    f1_upper = []
    for p, reps in provider_acceptance.items():
        for i in range(1, 4):
            if (p, i) in grouped:
                providers.append(f"{p} Rep {i}")
                b_ci = ci_map.get((p, i), {})
                f1_medians.append(b_ci.get("extraction_f1_median", 0))
                f1_lower.append(b_ci.get("extraction_f1_median", 0) - b_ci.get("extraction_f1_lower", 0))
                f1_upper.append(b_ci.get("extraction_f1_upper", 0) - b_ci.get("extraction_f1_median", 0))
    ax.errorbar(providers, f1_medians, yerr=[f1_lower, f1_upper], fmt='o')
    ax.set_title('Extraction F1 by Provider and Replicate')
    ax.set_ylabel('Extraction F1')
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig("results/provider_switch/figure_performance_ci.png")
    plt.close(fig)

    # Migration sequence
    fig, ax = plt.subplots(figsize=(8, 4))
    phases = []
    matches = []
    if os.path.exists("results/provider_switch/migration_hash_audit.csv"):
        with open("results/provider_switch/migration_hash_audit.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                phases.append(row["phase"])
                matches.append(1 if row["match"] == "True" else 0)
    ax.plot(range(len(phases)), matches, marker='o')
    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels(phases, rotation=45)
    ax.set_title('Structural Hash Audit State by Migration Phase')
    ax.set_ylabel('Match (1=True, 0=False)')
    plt.tight_layout()
    fig.savefig("results/provider_switch/figure_migration_sequence.png")
    plt.close(fig)

    # Provider agreement
    fig, ax = plt.subplots(figsize=(8, 6))
    comps = []
    diffs = []
    if os.path.exists("results/provider_switch/paired_provider_comparisons.csv"):
        with open("results/provider_switch/paired_provider_comparisons.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["metric"] == "extraction_f1":
                    comps.append(f"{row['provider_a']} vs {row['provider_b']} Rep {row['replicate']}")
                    diffs.append(float(row["difference"]))
    ax.bar(comps, diffs)
    ax.set_title('Paired Provider Agreement (Extraction F1 Difference)')
    ax.set_ylabel('Difference')
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig("results/provider_switch/figure_provider_agreement.png")
    plt.close(fig)

    hashes = {}
    artifacts = [
        "raw_outputs.jsonl", "case_level_results.csv", "replicate_summary.csv",
        "provider_summary.csv", "paired_provider_comparisons.csv", "noninferiority_results.csv",
        "authorization_results.csv", "provenance_results.csv", "migration_hash_audit.csv",
        "reversibility_results.csv", "acceptance_decisions.csv", "manuscript_results_blinded.json",
        "figure_performance_ci.png", "figure_migration_sequence.png", "figure_provider_agreement.png"
    ]
    for fname in artifacts:
        f = f"results/provider_switch/{fname}"
        if os.path.exists(f):
            h = hashlib.sha256()
            with open(f, "rb") as f2:
                while chunk := f2.read(8192): h.update(chunk)
            hashes[f] = h.hexdigest()
        
    with open("results/provider_switch/blinded_output_hashes.json", "w") as f:
        json.dump(hashes, f, indent=2)

if __name__ == "__main__":
    main()
