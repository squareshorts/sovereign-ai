"""
SPST Conformance Harness — Reference Implementation

Executes SPST components 1-6 against deterministic conformance fixtures.
This is a test harness, NOT a simulation or clinical validation tool.

Usage: python run_spst.py
"""

import sys
import os
import json
import csv
import hashlib
import shutil
import datetime
import importlib
import tempfile

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from workflow.task import (
    MedicationReconciliationTask, hash_data, reconcile_medications,
    ProvenanceRecord,
)
from workflow.schemas import validate_input, validate_output
from workflow.authorization import AuthorizationEngine
from adapters.conformant import ConformantFixture
from adapters.schema_failure import SchemaFailureFixture
from adapters.performance_failure import PerformanceFailureFixture
from adapters.provenance_failure import ProvenanceFailureFixture
from evaluation.synthetic_data import generate_evaluation_cases
from evaluation.metrics import compute_case_metrics, aggregate_metrics


# ── Institutional files: these must NOT change during migration ──────
INSTITUTIONAL_FILES = [
    "workflow_manifest.json",
    "workflow/__init__.py",
    "workflow/schemas.py",
    "workflow/authorization.py",
    "workflow/task.py",
    "evaluation/__init__.py",
    "evaluation/synthetic_data.py",
    "evaluation/metrics.py",
]


def hash_file(filepath):
    """SHA-256 hash of a file's contents."""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def hash_institutional_files():
    """Return dict of relative_path -> sha256 for all institutional files."""
    hashes = {}
    for relpath in INSTITUTIONAL_FILES:
        abspath = os.path.join(PROJECT_ROOT, relpath)
        if os.path.exists(abspath):
            hashes[relpath] = hash_file(abspath)
    return hashes


# ══════════════════════════════════════════════════════════════════════
# SPST-1: Exportability
# ══════════════════════════════════════════════════════════════════════
def run_spst1():
    print("\n" + "=" * 70)
    print("SPST-1: EXPORTABILITY")
    print("=" * 70)

    export_dir = os.path.join(PROJECT_ROOT, "results", "export_package")
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    os.makedirs(export_dir, exist_ok=True)

    # Copy all institutional files into versioned export package
    manifest_data = {}
    for relpath in INSTITUTIONAL_FILES:
        src = os.path.join(PROJECT_ROOT, relpath)
        dst = os.path.join(export_dir, relpath)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    # Generate content hashes for the export
    export_hashes = {}
    for relpath in INSTITUTIONAL_FILES:
        fpath = os.path.join(export_dir, relpath)
        if os.path.exists(fpath):
            export_hashes[relpath] = hash_file(fpath)

    # 1. Generate baseline output from original path
    manifest_path = os.path.join(PROJECT_ROOT, "workflow_manifest.json")
    with open(manifest_path, "r") as f:
        manifest_data = json.load(f)
    adapter = ConformantFixture()
    task = MedicationReconciliationTask(manifest_data, adapter)
    regression_cases = generate_evaluation_cases()[:1]
    case = regression_cases[0]
    baseline_output, _, _, _, _ = task.execute(case["case_id"], case["input"])

    # 2. Verify export package
    adapter_files_in_export = []
    for root, dirs, files in os.walk(export_dir):
        for fname in files:
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, export_dir)
            if rel.startswith("adapters"):
                adapter_files_in_export.append(rel)

    total_size = 0
    file_count = 0
    for root, dirs, files in os.walk(export_dir):
        for fname in files:
            total_size += os.path.getsize(os.path.join(root, fname))
            file_count += 1

    # 3. Clean reconstruction in isolated path
    recon_dir = os.path.join(PROJECT_ROOT, "results", "reconstruction")
    if os.path.exists(recon_dir):
        shutil.rmtree(recon_dir)
    shutil.copytree(export_dir, recon_dir)

    # Force a genuinely fresh import from the reconstructed package.
    # Changing sys.path alone is insufficient because "workflow.task" was
    # imported at module startup and would otherwise be returned from
    # sys.modules, silently defeating the isolation test.
    old_path = list(sys.path)
    module_prefixes = ("workflow", "evaluation")
    stashed_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name in module_prefixes
        or any(name.startswith(prefix + ".") for prefix in module_prefixes)
    }

    for name in stashed_modules:
        sys.modules.pop(name, None)

    if PROJECT_ROOT in sys.path:
        sys.path.remove(PROJECT_ROOT)
    sys.path.insert(0, recon_dir)

    reconstructed_output = None
    recon_manifest = {}
    recon_module_isolated = False

    try:
        recon_manifest_path = os.path.join(recon_dir, "workflow_manifest.json")
        with open(recon_manifest_path, "r") as f:
            recon_manifest = json.load(f)

        recon_task_module = importlib.import_module("workflow.task")
        recon_task_file = os.path.abspath(
            getattr(recon_task_module, "__file__", "")
        )
        recon_root = os.path.abspath(recon_dir) + os.sep
        recon_module_isolated = recon_task_file.startswith(recon_root)

        if not recon_module_isolated:
            raise RuntimeError(
                "Reconstruction isolation failed: workflow.task was loaded "
                f"from {recon_task_file!r}, not from {recon_dir!r}"
            )

        recon_task_class = recon_task_module.MedicationReconciliationTask

        # The adapter is intentionally injected from outside the export
        # package; SPST-1 tests portability of the institutional workflow layer.
        recon_task = recon_task_class(recon_manifest, ConformantFixture())
        reconstructed_output, _, _, _, _ = recon_task.execute(
            case["case_id"], case["input"]
        )
    finally:
        # Remove reconstruction modules, then restore the original module cache
        # and import path exactly as they were before this isolation test.
        for name in list(sys.modules):
            if (
                name in module_prefixes
                or any(
                    name.startswith(prefix + ".")
                    for prefix in module_prefixes
                )
            ):
                sys.modules.pop(name, None)

        sys.modules.update(stashed_modules)
        sys.path = old_path

    outputs_match = (
        reconstructed_output is not None
        and json.dumps(baseline_output, sort_keys=True)
        == json.dumps(reconstructed_output, sort_keys=True)
    )
    manifest_loadable = (
        recon_manifest.get("workflow", {}).get("id") == "med-recon-poc-v1"
    )

    passed = (
        len(adapter_files_in_export) == 0
        and manifest_loadable
        and recon_module_isolated
        and reconstructed_output is not None
        and outputs_match
    )

    result = {
        "files_exported": file_count,
        "package_size_bytes": total_size,
        "adapter_files_included": len(adapter_files_in_export),
        "manifest_loadable": manifest_loadable,
        "reconstructed_module_isolated": recon_module_isolated,
        "clean_reconstruction_successful": reconstructed_output is not None,
        "regression_output_equivalent": outputs_match,
        "pass": passed,
    }

    print(f"  Files exported:             {file_count}")
    print(f"  Package size:               {total_size} bytes")
    print(f"  Adapter files in export:    {len(adapter_files_in_export)}")
    print(f"  Reconstructed import clean: {recon_module_isolated}")
    print(f"  Clean reconstruction ok:    {result['clean_reconstruction_successful']}")
    print(f"  Regression equivalent:      {result['regression_output_equivalent']}")
    print(f"  SPST-1 RESULT:              {'PASS' if passed else 'FAIL'}")


    return result


# ══════════════════════════════════════════════════════════════════════
# SPST-2: Reconnection
# ══════════════════════════════════════════════════════════════════════
def run_spst2():
    print("\n" + "=" * 70)
    print("SPST-2: RECONNECTION")
    print("=" * 70)

    # Hash institutional files before switching
    hashes_before = hash_institutional_files()

    # 1. Baseline with config A
    manifest_path = os.path.join(PROJECT_ROOT, "workflow_manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    config_a_path = os.path.join(PROJECT_ROOT, "config", "deployment_a.json")
    with open(config_a_path, "r") as f:
        config_a = json.load(f)

    adapter_module_a = importlib.import_module(config_a["adapter_module"])
    adapter_class_a = getattr(adapter_module_a, config_a["adapter_class"])
    adapter_a = adapter_class_a()
    
    task_a = MedicationReconciliationTask(manifest, adapter_a)
    regression_cases = generate_evaluation_cases()[:1]
    case = regression_cases[0]
    
    output_a, prov_a, _, _, _ = task_a.execute(case["case_id"], case["input"])

    # 2. Switch to config B
    config_b_path = os.path.join(PROJECT_ROOT, "config", "deployment_b.json")
    with open(config_b_path, "r") as f:
        config_b = json.load(f)

    adapter_module_b = importlib.import_module(config_b["adapter_module"])
    adapter_class_b = getattr(adapter_module_b, config_b["adapter_class"])
    adapter_b = adapter_class_b()
    
    task_b = MedicationReconciliationTask(manifest, adapter_b)
    output_b, prov_b, _, _, _ = task_b.execute(case["case_id"], case["input"])

    # After "switching", hash institutional files again
    hashes_after = hash_institutional_files()

    # Compare
    changed_files = []
    for relpath in INSTITUTIONAL_FILES:
        h_before = hashes_before.get(relpath)
        h_after = hashes_after.get(relpath)
        if h_before != h_after:
            changed_files.append(relpath)

    # Count adapter-specific lines
    adapter_files = [
        "adapters/conformant.py",
        "adapters/schema_failure.py",
        "adapters/performance_failure.py",
        "adapters/provenance_failure.py",
        "adapters/base.py",
    ]
    adapter_loc = 0
    for af in adapter_files:
        fpath = os.path.join(PROJECT_ROOT, af)
        if os.path.exists(fpath):
            with open(fpath, "r") as f:
                adapter_loc += len(f.readlines())

    institutional_hash_equal = len(changed_files) == 0
    b_instantiated = prov_b.fixture_id == config_b["active_fixture"]
    
    passed = institutional_hash_equal and b_instantiated

    result = {
        "institutional_files_changed": len(changed_files),
        "changed_files": changed_files,
        "institutional_hash_equal": institutional_hash_equal,
        "adapter_specific_loc": adapter_loc,
        "config_a": config_a,
        "config_b": config_b,
        "b_instantiated": b_instantiated,
        "pass": passed,
    }

    print(f"  Institutional files changed: {len(changed_files)}")
    print(f"  Institutional hash equal:    {institutional_hash_equal}")
    print(f"  Adapter-specific LOC:        {adapter_loc}")
    print(f"  Config A fixture:            {config_a['active_fixture']}")
    print(f"  Config B fixture:            {config_b['active_fixture']}")
    print(f"  Adapter B reached:           {b_instantiated}")
    print(f"  SPST-2 RESULT:               {'PASS' if passed else 'FAIL'}")

    return result


# ══════════════════════════════════════════════════════════════════════
# SPST-3: Authorization Preservation
# ══════════════════════════════════════════════════════════════════════
def run_spst3(manifest, fixtures):
    print("\n" + "=" * 70)
    print("SPST-3: AUTHORIZATION PRESERVATION")
    print("=" * 70)

    cases = generate_evaluation_cases()
    auth_cases = [c for c in cases if c["authorization_test"]]
    results = []

    for fixture_name, adapter in fixtures.items():
        task = MedicationReconciliationTask(manifest, adapter)
        engine = AuthorizationEngine(manifest)

        for case in auth_cases:
            res = task.execute(
                case["case_id"], case["input"]
            )

            # Also test resource gate operations
            gate_write = engine.gate_operation(
                "write", "synthetic_patient_records")
            gate_delete = engine.gate_operation(
                "delete", "synthetic_patient_records")

            record = {
                "case_id": case["case_id"],
                "fixture": fixture_name,
                "input_auth_attempted": auth_result.attempted,
                "input_auth_blocked": auth_result.blocked,
                "input_auth_executed": auth_result.executed,
                "gate_write_blocked": gate_write.blocked,
                "gate_delete_blocked": gate_delete.blocked,
                "output_produced": output is not None,
                "details": auth_result.details,
            }
            results.append(record)

    # Check: all auth cases should have attempted=True, blocked=True, executed=False
    all_blocked = all(
        r["input_auth_attempted"] and
        r["input_auth_blocked"] and
        not r["input_auth_executed"] and
        not r["output_produced"]
        for r in results
    )
    # All gate ops should block write and delete
    all_gates_blocked = all(
        r["gate_write_blocked"] and r["gate_delete_blocked"]
        for r in results
    )

    passed = all_blocked and all_gates_blocked

    print(f"  Authorization test cases:    {len(auth_cases)}")
    print(f"  Fixtures tested:             {len(fixtures)}")
    print(f"  Total negative tests:        {len(results)}")
    print(f"  All prohibited ops blocked:  {all_blocked}")
    print(f"  All resource gates blocked:  {all_gates_blocked}")
    print(f"  SPST-3 RESULT:               {'PASS' if passed else 'FAIL'}")

    return {
        "test_count": len(results),
        "all_blocked": all_blocked,
        "all_gates_blocked": all_gates_blocked,
        "results": results,
        "pass": passed,
    }


# ══════════════════════════════════════════════════════════════════════
# SPST-4: Provenance Preservation
# ══════════════════════════════════════════════════════════════════════
def run_spst4(manifest, fixtures):
    print("\n" + "=" * 70)
    print("SPST-4: PROVENANCE PRESERVATION")
    print("=" * 70)

    cases = generate_evaluation_cases()
    non_auth_cases = [c for c in cases if not c["authorization_test"]]
    provenance_records = []
    validation_results = []

    for fixture_name, adapter in fixtures.items():
        task = MedicationReconciliationTask(manifest, adapter)
        for case in non_auth_cases[:10]:  # Subset for provenance test
            res = task.execute(
                case["case_id"], case["input"]
            )
            prov_dict = prov.to_dict()
            ok, prov_errs = prov.validate_completeness(manifest)
            provenance_records.append(prov_dict)
            validation_results.append({
                "case_id": case["case_id"],
                "fixture": fixture_name,
                "complete": ok,
                "errors": prov_errs,
            })

    total = len(validation_results)
    complete_count = sum(1 for v in validation_results if v["complete"])
    completeness_rate = complete_count / total if total > 0 else 0

    conformant_complete = next((v["complete"] for v in validation_results if v["fixture"] == "conformant_fixture"), False)
    schema_complete = next((v["complete"] for v in validation_results if v["fixture"] == "schema_failure_fixture"), False)
    perf_complete = next((v["complete"] for v in validation_results if v["fixture"] == "performance_failure_fixture"), False)
    prov_complete = next((v["complete"] for v in validation_results if v["fixture"] == "provenance_failure_fixture"), True)

    machinery_works = (
        conformant_complete and 
        schema_complete and 
        perf_complete and 
        not prov_complete
    )

    return {
        "total_records": total,
        "complete_count": complete_count,
        "completeness_rate": completeness_rate,
        "validation_results": validation_results,
        "sample_provenance": provenance_records[:2] if provenance_records else [],
        "pass": machinery_works,
    }


# ══════════════════════════════════════════════════════════════════════
# SPST-5: Performance Acceptance (machinery test only)
# ══════════════════════════════════════════════════════════════════════
def run_spst5(manifest, fixtures):
    print("\n" + "=" * 70)
    print("SPST-5: PERFORMANCE ACCEPTANCE (machinery test)")
    print("NOTE: Substantive performance portability was NOT evaluated")
    print("      because no independent real inference models were available.")
    print("=" * 70)

    cases = generate_evaluation_cases()
    thresholds = manifest["evaluation"]["acceptance_thresholds"]
    all_results_csv = []  # For CSV output
    fixture_summaries = {}

    def check_acceptance_envelope(agg):
        if agg['schema_valid_rate'] < thresholds['schema_valid_rate']: return False
        if agg['discrepancy_sensitivity'] is not None and agg['discrepancy_sensitivity'] < thresholds['discrepancy_sensitivity']: return False
        if agg['discrepancy_precision'] is not None and agg['discrepancy_precision'] < thresholds['discrepancy_precision']: return False
        if agg['human_review_sensitivity'] is not None and agg['human_review_sensitivity'] < thresholds['human_review_sensitivity']: return False
        if agg['critical_error_rate'] > thresholds['critical_error_rate_max']: return False
        return True

    for fixture_name, adapter in fixtures.items():
        task = MedicationReconciliationTask(manifest, adapter)
        case_metrics = []

        for case in cases:
            res = task.execute(case["case_id"], case["input"])
            
            output = res.extracted_output
            prov = res.provenance
            auth_result = res.authorization
            schema_ok = res.schema_valid
            
            prov_ok, _ = prov.validate_completeness(manifest)

            metrics = compute_case_metrics(
                output=output,
                ground_truth=case["ground_truth"],
                has_discrepancy=case["has_discrepancy"],
                human_review_expected=case["human_review_expected"],
                schema_valid=schema_ok,
                authorization_test=case["authorization_test"],
                auth_blocked=auth_result.blocked,
            )
            case_metrics.append(metrics)

            # Build CSV row
            csv_row = {
                "case_id": case["case_id"],
                "fixture": fixture_name,
                "model": adapter.MODEL_ID,
                "schema_valid": int(schema_ok),
                "discrepancy_tp": metrics["discrepancy_tp"],
                "discrepancy_fp": metrics["discrepancy_fp"],
                "discrepancy_fn": metrics["discrepancy_fn"],
                "discrepancy_tn": metrics["discrepancy_tn"],
                "human_review_tp": metrics["human_review_tp"],
                "human_review_fp": metrics["human_review_fp"],
                "human_review_fn": metrics["human_review_fn"],
                "human_review_tn": metrics["human_review_tn"],
                "critical_error": metrics["critical_error"],
                "authorization_test": int(case["authorization_test"]),
                "auth_blocked": int(auth_result.blocked),
                "provenance_complete": int(prov_ok),
            }
            all_results_csv.append(csv_row)

        agg = aggregate_metrics(case_metrics)
        fixture_summaries[fixture_name] = agg

        print(f"\n  Fixture: {fixture_name}")
        print(f"    Cases evaluated:           {agg['n_cases']}")
        print(f"    Schema valid rate:         {agg['schema_valid_rate']:.2%}")
        print(f"    Discrepancy TP/FP/FN/TN:   "
              f"{agg['discrepancy_tp']}/{agg['discrepancy_fp']}/"
              f"{agg['discrepancy_fn']}/{agg['discrepancy_tn']}")
        if agg['discrepancy_sensitivity'] is not None:
            print(f"    Discrepancy sensitivity:   {agg['discrepancy_sensitivity']:.2%}")
        if agg['discrepancy_precision'] is not None:
            print(f"    Discrepancy precision:     {agg['discrepancy_precision']:.2%}")
        if agg['human_review_sensitivity'] is not None:
            print(f"    Human-review sensitivity:  {agg['human_review_sensitivity']:.2%}")
        print(f"    Critical error rate:       {agg['critical_error_rate']:.2%}")

        # Check thresholds
        passes_threshold = check_acceptance_envelope(agg)
        print(f"    Meets acceptance envelope:  {passes_threshold}")

    # Write CSV
    csv_path = os.path.join(PROJECT_ROOT, "results", "spst_results.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if all_results_csv:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results_csv[0].keys())
            writer.writeheader()
            writer.writerows(all_results_csv)
        print(f"\n  CSV written: {csv_path} ({len(all_results_csv)} rows)")

    conformant_passes = False
    schema_passes = True
    perf_passes = True
    
    if "conformant_fixture" in fixture_summaries:
        conformant_passes = check_acceptance_envelope(fixture_summaries["conformant_fixture"])
        
    if "schema_failure_fixture" in fixture_summaries:
        schema_passes = check_acceptance_envelope(fixture_summaries["schema_failure_fixture"])
        
    if "performance_failure_fixture" in fixture_summaries:
        perf_passes = check_acceptance_envelope(fixture_summaries["performance_failure_fixture"])

    machinery_works = conformant_passes and not schema_passes and not perf_passes

    print(f"\n  SPST-5 RESULT: {'PASS' if machinery_works else 'FAIL'} "
          f"(acceptance-envelope machinery functions correctly)")

    return {
        "fixture_summaries": fixture_summaries,
        "csv_rows": len(all_results_csv),
        "thresholds": thresholds,
        "pass": machinery_works,
        "note": ("Substantive performance portability was NOT evaluated "
                 "because no independent real inference models were available."),
    }


# ══════════════════════════════════════════════════════════════════════
# SPST-6: Reversibility (A -> B -> A rollback)
# ══════════════════════════════════════════════════════════════════════
def run_spst6(manifest):
    print("\n" + "=" * 70)
    print("SPST-6: REVERSIBILITY (A -> B -> A)")
    print("=" * 70)

    active_config_path = os.path.join(PROJECT_ROOT, "config", "active_deployment.json")
    config_a_path = os.path.join(PROJECT_ROOT, "config", "deployment_a.json")
    config_b_path = os.path.join(PROJECT_ROOT, "config", "deployment_b.json")
    
    def set_active_config(src_path):
        shutil.copy2(src_path, active_config_path)
    
    def load_active_task():
        with open(active_config_path, "r") as f:
            cfg = json.load(f)
        mod = importlib.import_module(cfg["adapter_module"])
        adapter = getattr(mod, cfg["adapter_class"])()
        return MedicationReconciliationTask(manifest, adapter)

    # --- Phase 1: Baseline with fixture A ---
    print("  Phase 1: Active configuration A loaded")
    set_active_config(config_a_path)
    hashes_baseline = hash_institutional_files()
    task_a = load_active_task()

    regression_cases = generate_evaluation_cases()[:5]
    baseline_outputs = []
    for case in regression_cases:
        output, prov, auth, schema_ok, errs = task_a.execute(
            case["case_id"], case["input"]
        )
        prov_ok, _ = prov.validate_completeness(manifest)
        baseline_outputs.append({
            "case_id": case["case_id"],
            "output": output,
            "output_hash": prov.output_hash,
            "schema_ok": schema_ok,
            "prov_ok": prov_ok,
        })

    # --- Phase 2: Switch to fixture B ---
    print("  Phase 2: Switch to B confirmed")
    set_active_config(config_b_path)
    hashes_after_switch = hash_institutional_files()
    task_b = load_active_task()

    switch_outputs = []
    for case in regression_cases:
        output, prov, auth, schema_ok, errs = task_b.execute(
            case["case_id"], case["input"]
        )
        prov_ok, _ = prov.validate_completeness(manifest)
        switch_outputs.append({
            "case_id": case["case_id"],
            "output": output,
            "output_hash": prov.output_hash,
            "schema_ok": schema_ok,
            "prov_ok": prov_ok,
        })

    # --- Phase 3: Configuration restored to A ---
    print("  Phase 3: Configuration restored to A")
    set_active_config(config_a_path)
    hashes_after_rollback = hash_institutional_files()
    task_a_rollback = load_active_task()

    rollback_outputs = []
    for case in regression_cases:
        output, prov, auth, schema_ok, errs = task_a_rollback.execute(
            case["case_id"], case["input"]
        )
        prov_ok, _ = prov.validate_completeness(manifest)
        rollback_outputs.append({
            "case_id": case["case_id"],
            "output": output,
            "output_hash": prov.output_hash,
            "schema_ok": schema_ok,
            "prov_ok": prov_ok,
        })

    # --- Verification ---
    with open(config_a_path, "r") as f:
        orig_a = json.load(f)
    with open(active_config_path, "r") as f:
        restored_a = json.load(f)

    config_restored = (orig_a == restored_a)
    institutional_hash_match = (hashes_baseline == hashes_after_rollback)
    validated_config_match = (manifest["rollback"]["validated_configuration"] == restored_a["active_fixture"])

    # Check regression: baseline outputs should match rollback outputs
    regression_match = all(
        b["output_hash"] == r["output_hash"]
        for b, r in zip(baseline_outputs, rollback_outputs)
    )

    # Provenance continuity: all three phases produced complete provenance
    provenance_continuous = True
    for phase_outputs in [baseline_outputs, switch_outputs, rollback_outputs]:
        for res in phase_outputs:
            if not res["prov_ok"]:
                provenance_continuous = False

    passed = config_restored and institutional_hash_match and regression_match and provenance_continuous and validated_config_match

    print(f"  Restored config == baseline:           {config_restored}")
    print(f"  Validated config matches manifest:     {validated_config_match}")
    print(f"  0 institutional files changed:         {institutional_hash_match}")
    print(f"  Regression outputs A-before == A-after:{regression_match}")
    print(f"  Provenance continuity:                 {provenance_continuous}")
    print(f"  SPST-6 RESULT:                         {'PASS' if passed else 'FAIL'}")

    return {
        "config_restored": config_restored,
        "institutional_hash_match": institutional_hash_match,
        "regression_match": regression_match,
        "provenance_continuous": provenance_continuous,
        "hashes_baseline": hashes_baseline,
        "hashes_after_switch": hashes_after_switch,
        "hashes_after_rollback": hashes_after_rollback,
        "institutional_files_changed_during_switch": sum(
            1 for k in hashes_baseline
            if hashes_baseline[k] != hashes_after_switch.get(k)
        ),
        "institutional_files_changed_during_rollback": sum(
            1 for k in hashes_baseline
            if hashes_baseline[k] != hashes_after_rollback.get(k)
        ),
        "pass": passed,
    }


# ══════════════════════════════════════════════════════════════════════
# Main orchestrator
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("SPST REFERENCE IMPLEMENTATION -- CONFORMANCE HARNESS")
    print(f"Execution timestamp: {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    print("=" * 70)
    print("\nThis harness evaluates the SPST test machinery using")
    print("deterministic conformance fixtures. It does NOT establish")
    print("behavioral portability across real models or providers.")

    # Load manifest
    manifest_path = os.path.join(PROJECT_ROOT, "workflow_manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Define fixtures
    fixtures = {
        "conformant_fixture": ConformantFixture(),
        "schema_failure_fixture": SchemaFailureFixture(),
        "performance_failure_fixture": PerformanceFailureFixture(),
        "provenance_failure_fixture": ProvenanceFailureFixture(),
    }

    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)

    # Run all SPST components
    spst1 = run_spst1()
    spst2 = run_spst2()
    spst3 = run_spst3(manifest, fixtures)
    spst4 = run_spst4(manifest, fixtures)
    spst5 = run_spst5(manifest, fixtures)
    spst6 = run_spst6(manifest)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SPST COMPONENT SUMMARY")
    print("=" * 70)

    components = [
        ("SPST-1: Exportability", spst1["pass"], "Institutional package exported without adapter state"),
        ("SPST-2: Reconnection", spst2["pass"], f"0 institutional files changed; {spst2['adapter_specific_loc']} adapter LOC"),
        ("SPST-3: Authorization", spst3["pass"], f"{spst3['test_count']} negative tests, all prohibited ops blocked"),
        ("SPST-4: Provenance", spst4["pass"], f"{spst4['complete_count']}/{spst4['total_records']} records complete (machinery detects gaps)"),
        ("SPST-5: Performance", spst5["pass"], "Acceptance-envelope machinery functions correctly (no real models tested)"),
        ("SPST-6: Reversibility", spst6["pass"], "A->B->A rollback verified, regression outputs match"),
    ]

    summary_data = []
    for name, passed, evidence in components:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:40s} {status:6s}  {evidence}")
        summary_data.append({
            "component": name,
            "result": status,
            "evidence": evidence,
        })

    # Write summary JSON
    summary = {
        "execution_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "harness_type": "reference_implementation_conformance_test",
        "disclaimer": (
            "This harness uses deterministic mock fixtures. "
            "It does NOT establish behavioral portability across "
            "real foundation models or organizational providers."
        ),
        "components": summary_data,
        "spst1_detail": {k: v for k, v in spst1.items() if k != "export_hashes"},
        "spst2_detail": {k: v for k, v in spst2.items()
                         if k not in ("hashes_before", "hashes_after")},
        "spst3_detail": {"test_count": spst3["test_count"],
                         "all_blocked": spst3["all_blocked"],
                         "all_gates_blocked": spst3["all_gates_blocked"],
                         "pass": spst3["pass"]},
        "spst4_detail": {"total_records": spst4["total_records"],
                         "complete_count": spst4["complete_count"],
                         "completeness_rate": spst4["completeness_rate"],
                         "pass": spst4["pass"]},
        "spst5_detail": {
            "fixture_summaries": {
                k: {mk: mv for mk, mv in v.items()}
                for k, v in spst5["fixture_summaries"].items()
            },
            "note": spst5["note"],
            "pass": spst5["pass"],
        },
        "spst6_detail": {
            "config_restored": spst6["config_restored"],
            "institutional_hash_match": spst6["institutional_hash_match"],
            "regression_match": spst6["regression_match"],
            "provenance_continuous": spst6["provenance_continuous"],
            "institutional_files_changed_during_switch": spst6["institutional_files_changed_during_switch"],
            "institutional_files_changed_during_rollback": spst6["institutional_files_changed_during_rollback"],
            "pass": spst6["pass"],
        },
    }

    summary_path = os.path.join(PROJECT_ROOT, "results", "summary_results.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary written: {summary_path}")

    overall_pass = all(c["result"] == "PASS" for c in summary_data)
    print(f"\n  Overall: {'ALL PASS' if overall_pass else 'SOME FAILURES'}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
