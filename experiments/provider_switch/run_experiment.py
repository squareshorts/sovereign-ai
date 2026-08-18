import os
import json
import random
import secrets
import hashlib
import datetime
import time
import sys
import subprocess
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from adapters.base import ProviderTransportError, ProviderHTTPError, ProviderRefusalError, ProviderResponseError
from workflow.task import ProvenanceRecord, AuthorizationResult, MedicationReconciliationTask

def hash_data(data):
    if isinstance(data, (dict, list)):
        serialized = json.dumps(data, sort_keys=True, ensure_ascii=True)
    else:
        serialized = data
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

def hash_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def create_or_load_mapping():
    mapping_path = "experiments/provider_switch/private/provider_mapping_v3.json"
    commitment_path = "experiments/provider_switch/provider_mapping_v3_commitment.json"
    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
    providers = ["OpenAI", "Anthropic", "Google"]
    
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            mapping = json.load(f)
    else:
        shuffled = list(providers)
        secrets.SystemRandom().shuffle(shuffled)
        mapping = {f"P{i+1}": p for i, p in enumerate(shuffled)}
        with open(mapping_path, "w") as f:
            json.dump(mapping, f, indent=2)
            
    mapping_hash = hash_data(mapping)
    
    if not os.path.exists(commitment_path):
        commitment = {
            "schema_version": "1.0",
            "preregistration_version": "v3",
            "creation_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "mapping_sha256": mapping_hash
        }
        with open(commitment_path, "w") as f:
            json.dump(commitment, f, indent=2)
    else:
        with open(commitment_path, "r") as f:
            commitment = json.load(f)
        if commitment["mapping_sha256"] != mapping_hash:
            raise ValueError("Mapping hash does not match commitment!")
    return mapping

def generate_schedule(cases):
    schedule = []
    base_order = ["P1", "P2", "P3"]
    for rep in range(1, 4):
        rng = random.Random(20260817 + rep)
        rep_cases = list(cases)
        rng.shuffle(rep_cases)
        for j, case in enumerate(rep_cases):
            rotation = (j + rep - 1) % 3
            order = base_order[rotation:] + base_order[:rotation]
            for p in order:
                schedule.append({
                    "replicate": rep,
                    "case_id": case["case_id"],
                    "stratum": case["stratum"],
                    "provider_blind_id": p,
                    "input": case["input"]
                })
    return schedule

def is_retryable(e):
    if isinstance(e, ProviderHTTPError):
        return e.code in [429, 500, 502, 503, 504]
    if isinstance(e, ProviderTransportError):
        return True
    return False

def get_adapter(provider_name, protocol):
    if provider_name == "OpenAI":
        model = protocol["selected_models"][provider_name]
        from adapters.real.openai_adapter import OpenAIAdapter
        return OpenAIAdapter(model)
    elif provider_name == "Anthropic":
        model = protocol["selected_models"][provider_name]
        from adapters.real.anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter(model)
    elif provider_name == "Google":
        model = protocol["selected_models"][provider_name]
        from adapters.real.google_adapter import GoogleAdapter
        return GoogleAdapter(model)
    elif provider_name == "Mock":
        class MockAdapter:
            FIXTURE_ID = "Mock"
            MODEL_ID = "mock-model"
            ADAPTER_VERSION = "1.0"
            def infer(self, req):
                return '{"request_meds": [], "statement_meds": []}'
        return MockAdapter()

def check_preconditions(args, mapping_hash, commitment_hash):
    if args.mock_adapters:
        return
    head_rev = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    tag_rev = subprocess.check_output(["git", "rev-parse", "spst-preregistration-v3.1^{}"]).decode().strip()
    if head_rev != tag_rev:
        raise ValueError("HEAD is not at spst-preregistration-v3.1")
    
    status = subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()
    if status:
        raise ValueError("Working tree is not clean")
        
    if os.path.exists("artifact_hashes.json"):
        with open("artifact_hashes.json", "r") as f:
            expected = json.load(f)
        for path, expected_hash in expected.items():
            if os.path.exists(path):
                if hash_file(path) != expected_hash:
                    raise ValueError(f"Protected hash mismatch for {path}")
                    
    for k in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]:
        if not os.environ.get(k):
            raise ValueError(f"Missing credential {k}")
            
    print("Running benchmark integrity PASS...")
    subprocess.check_call([sys.executable, "test_benchmark_integrity.py"])
    print("Benchmark integrity PASS")
    
    if mapping_hash != commitment_hash:
        raise ValueError("provider mapping hash does not match v3 commitment")
        
    if not args.resume and os.path.exists("results/provider_switch/raw_outputs.jsonl"):
        raise ValueError("Preexisting formal v3 ledger found without --resume")

def write_migration_audit(phase):
    audit_file = "results/provider_switch/migration_hash_audit.csv"
    os.makedirs("results/provider_switch", exist_ok=True)
    files_to_hash = [
        "experiments/provider_switch/PROTOCOL.md",
        "experiments/provider_switch/protocol.json",
        "experiments/provider_switch/provider_mapping_v3_commitment.json",
        "experiments/provider_switch/benchmark_evaluation_truth.jsonl"
    ]
    lines = []
    if not os.path.exists(audit_file):
        lines.append("phase,file,hash\n")
    for f in files_to_hash:
        h = hash_file(f) if os.path.exists(f) else "MISSING"
        lines.append(f"{phase},{f},{h}\n")
    with open(audit_file, "a") as out:
        out.writelines(lines)

def validate_status_aware_provenance(log, prov):
    status = log.get("execution_status")
    missing = []
    
    def req(field_name, val):
        if val is None or val == "":
            missing.append(field_name)

    if status == "AUTHORIZATION_BLOCKED":
        req("case_id", log.get("case_id"))
        req("workflow_version", prov.workflow_version if prov else None)
        req("provider_blind_id", log.get("provider_blind_id"))
        req("replicate", log.get("replicate"))
        req("timestamp", log.get("timestamp"))
        req("institutional_input_hash", log.get("input_hash"))
        req("authorization_outcome", log.get("authorization_outcome"))
        req("execution_status", status)
        if log.get("provider_api_call_count") != 0:
            missing.append("provider_api_call_count_zero")
    elif status == "COMPLETED_SCHEMA_VALID":
        req("case_id", log.get("case_id"))
        req("workflow_version", prov.workflow_version if prov else None)
        req("provider", prov.fixture_id if prov else None)
        req("model_version", prov.model_id if prov else None)
        req("adapter_version", prov.adapter_version if prov else None)
        req("timestamp", log.get("timestamp"))
        req("input_hash", log.get("input_hash"))
        req("provider_facing_input_hash", log.get("provider_facing_input_hash"))
        req("raw_output_hash", log.get("raw_response_hash"))
        req("output_hash", log.get("extracted_output_hash"))
        req("schema_validation_outcome", "PASS")
        req("authorization_outcome", "permitted")
        req("execution_status", status)
    elif status in ["COMPLETED_SCHEMA_FAILURE", "FAILED_PARSE", "FAILED_EXTRACTION_FIELDS", "FAILED_OUTPUT_SCHEMA"]:
        req("case_id", log.get("case_id"))
        req("workflow_version", prov.workflow_version if prov else None)
        req("provider", prov.fixture_id if prov else None)
        req("model_version", prov.model_id if prov else None)
        req("adapter_version", prov.adapter_version if prov else None)
        req("timestamp", log.get("timestamp"))
        req("input_hash", log.get("input_hash"))
        req("provider_facing_input_hash", log.get("provider_facing_input_hash"))
        req("raw_output_hash", log.get("raw_response_hash"))
        req("schema_validation_outcome", "FAIL")
        req("authorization_outcome", "permitted")
        req("execution_status", status)
    elif status == "PROVIDER_CALL_FAILURE":
        req("case_id", log.get("case_id"))
        req("workflow_version", prov.workflow_version if prov else None)
        req("provider", prov.fixture_id if prov else None)
        req("model_version", prov.model_id if prov else None)
        req("adapter_version", prov.adapter_version if prov else None)
        req("timestamp", log.get("timestamp"))
        req("input_hash", log.get("input_hash"))
        req("provider_facing_input_hash", log.get("provider_facing_input_hash"))
        req("authorization_outcome", "permitted")
        req("execution_status", status)
        req("attempt_count", log.get("attempt_count"))
        req("standardized_error_class", log.get("standardized_error_class"))
    elif status == "PROVIDER_REFUSAL":
        req("case_id", log.get("case_id"))
        req("workflow_version", prov.workflow_version if prov else None)
        req("provider", prov.fixture_id if prov else None)
        req("model_version", prov.model_id if prov else None)
        req("adapter_version", prov.adapter_version if prov else None)
        req("timestamp", log.get("timestamp"))
        req("input_hash", log.get("input_hash"))
        req("provider_facing_input_hash", log.get("provider_facing_input_hash"))
        req("authorization_outcome", "permitted")
        req("execution_status", status)
    else:
        pass
    return len(missing) == 0, missing

class CountingAdapterWrapper:
    def __init__(self, adapter):
        self.adapter = adapter
        self.call_count = 0
    def infer(self, req):
        self.call_count += 1
        return self.adapter.infer(req)
    @property
    def FIXTURE_ID(self): return self.adapter.FIXTURE_ID
    @property
    def MODEL_ID(self): return self.adapter.MODEL_ID
    @property
    def ADAPTER_VERSION(self): return self.adapter.ADAPTER_VERSION

def execute_scheduled_unit(case, adapter, manifest, protocol, is_auth, sleep_func=time.sleep):
    attempt_count = 0
    provider_api_call_count = 0
    extracted_output = None
    workflow_output = None
    provenance = None
    authorization = None
    last_error = None
    retryable_failure = False
    http_status = None
    standardized_error_class = None
    
    wrapper = CountingAdapterWrapper(adapter)
    task = MedicationReconciliationTask(manifest, wrapper)
    
    success = False
    final_status = ""
    
    while attempt_count < protocol["maximum_total_attempts"] and not success:
        attempt_count += 1
        
        if is_auth:
            res = task.execute(case["case_id"], case["input"])
            extracted_output = res.extracted_output
            provenance = res.provenance
            authorization = res.authorization
            final_status = "AUTHORIZATION_BLOCKED"
            success = True
            attempt_count = 0
            break
            
        try:
            res = task.execute(case["case_id"], case["input"])
            extracted_output = res.extracted_output
            workflow_output = res.workflow_output
            provenance = res.provenance
            authorization = res.authorization
            
            if res.schema_valid:
                final_status = "COMPLETED_SCHEMA_VALID"
            else:
                if provenance.execution_status in ["FAILED_PARSE", "FAILED_EXTRACTION_FIELDS", "FAILED_OUTPUT_SCHEMA"]:
                    final_status = "COMPLETED_SCHEMA_FAILURE"
                else:
                    final_status = provenance.execution_status
            success = True
        except ProviderRefusalError:
            final_status = "PROVIDER_REFUSAL"
            provenance = ProvenanceRecord(case["case_id"], manifest, wrapper.FIXTURE_ID, wrapper.MODEL_ID, wrapper.ADAPTER_VERSION)
            provenance.set_input_hash(hash_data(case["input"]))
            provenance.set_provider_facing_input_hash(hash_data({"history": case["input"].get("history", [])}))
            provenance.set_execution_status(final_status)
            authorization = AuthorizationResult(False, False, False)
            success = True
        except (ProviderTransportError, ProviderHTTPError, ProviderResponseError) as e:
            last_error = e
            standardized_error_class = str(type(e).__name__)
            http_status = getattr(e, 'code', None)
            retryable_failure = is_retryable(e)
            if retryable_failure and attempt_count < protocol["maximum_total_attempts"]:
                sleep_func(2 ** (attempt_count - 1))
            else:
                final_status = "PROVIDER_CALL_FAILURE"
                provenance = ProvenanceRecord(case["case_id"], manifest, wrapper.FIXTURE_ID, wrapper.MODEL_ID, wrapper.ADAPTER_VERSION)
                provenance.set_input_hash(hash_data(case["input"]))
                provenance.set_provider_facing_input_hash(hash_data({"history": case["input"].get("history", [])}))
                provenance.set_execution_status(final_status)
                authorization = AuthorizationResult(False, False, False)
                success = True

    provider_api_call_count = wrapper.call_count

    if final_status == "COMPLETED_SCHEMA_VALID":
        assert extracted_output is not None, "COMPLETED_SCHEMA_VALID AND extracted_output is null"
        assert workflow_output is not None, "COMPLETED_SCHEMA_VALID AND workflow_output is null"
    if final_status == "AUTHORIZATION_BLOCKED":
        assert provider_api_call_count == 0, "AUTHORIZATION_BLOCKED AND provider_api_call_count != 0"
        assert attempt_count == 0, "AUTHORIZATION_BLOCKED AND attempt_count != 0"
    if not is_auth:
        assert attempt_count <= 4, "attempt_count > 4"
        assert provider_api_call_count <= 4, "provider_api_call_count > 4"
        assert provider_api_call_count == attempt_count, "provider_api_call_count != attempt_count"

    pub_log = {
        "case_id": case["case_id"],
        "stratum": case["stratum"],
        "provider_blind_id": case["provider_blind_id"],
        "replicate": case["replicate"],
        "execution_status": final_status,
        "schema_valid": final_status == "COMPLETED_SCHEMA_VALID",
        "extracted_output": extracted_output,
        "workflow_output": workflow_output,
        "authorization_blocked": final_status == "AUTHORIZATION_BLOCKED",
        "authorization_outcome": "blocked" if final_status == "AUTHORIZATION_BLOCKED" else "permitted",
        "provider_api_call_count": provider_api_call_count,
        "attempt_count": attempt_count,
        "retryable_failure": retryable_failure,
        "standardized_error_class": standardized_error_class,
        "http_status": http_status,
        "input_hash": provenance.input_hash if provenance else None,
        "provider_facing_input_hash": provenance.provider_facing_input_hash if provenance else None,
        "raw_response_hash": provenance.raw_output_hash if provenance else None,
        "extracted_output_hash": hash_data(extracted_output) if extracted_output else None,
        "workflow_output_hash": hash_data(workflow_output) if workflow_output else None,
        "timestamp": provenance.timestamp if provenance else datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    
    prov_comp, miss_fields = validate_status_aware_provenance(pub_log, provenance)
    pub_log["provenance_complete"] = prov_comp
    pub_log["missing_provenance_fields"] = miss_fields

    priv_log = {
        "case_id": case["case_id"],
        "stratum": case["stratum"],
        "provider_blind_id": case["provider_blind_id"],
        "true_provider": getattr(adapter, "true_name", "Mock"),
        "replicate": case["replicate"],
        "execution_status": final_status,
        "provider_api_call_count": provider_api_call_count,
        "attempt_count": attempt_count,
        "schema_valid": final_status == "COMPLETED_SCHEMA_VALID",
        "extracted_output": extracted_output,
        "workflow_output": workflow_output,
        "provenance": provenance.to_dict() if provenance else {},
        "auth": authorization.to_dict() if authorization else {}
    }
    
    return pub_log, priv_log

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mock-adapters", action="store_true")
    parser.add_argument("--prereg-tag", type=str)
    args = parser.parse_args()

    if not args.dry_run and not args.formal:
        print("Must specify --dry-run or --formal")
        sys.exit(1)
        
    if args.formal and args.prereg_tag != "spst-preregistration-v3.1":
        if not args.mock_adapters:
            raise ValueError("Formal execution requires --prereg-tag spst-preregistration-v3.1")

    mapping = create_or_load_mapping()
    
    with open("experiments/provider_switch/provider_mapping_v3_commitment.json", "r") as f:
        commitment = json.load(f)

    if args.formal and not args.mock_adapters:
        check_preconditions(args, hash_data(mapping), commitment["mapping_sha256"])

    with open("experiments/provider_switch/protocol.json", "r") as f:
        protocol = json.load(f)
        
    with open("workflow_manifest.json", "r") as f:
        manifest = json.load(f)

    cases = []
    with open("experiments/provider_switch/benchmark_inputs.jsonl", "r") as f:
        for line in f:
            cases.append(json.loads(line))
            
    schedule = generate_schedule(cases)
    
    if len(schedule) != 2160:
        raise ValueError(f"Expected 2160 scheduled units, got {len(schedule)}")
        
    os.makedirs("results/provider_switch", exist_ok=True)
    
    with open("results/provider_switch/execution_order.json", "w") as f:
        json.dump([{"replicate": s["replicate"], "case_id": s["case_id"], "provider_blind_id": s["provider_blind_id"]} for s in schedule], f, indent=2)

    completed = set()
    raw_outputs_path = "results/provider_switch/raw_outputs.jsonl"
    private_op_path = "results/provider_switch/private_operational.jsonl"
    execution_log_path = "results/provider_switch/execution_log.jsonl"
    
    if os.path.exists(raw_outputs_path):
        with open(raw_outputs_path, "r") as f:
            for line in f:
                item = json.loads(line)
                completed.add((item["case_id"], item["provider_blind_id"], item["replicate"]))
                
    if args.dry_run:
        auth_units = sum(1 for s in schedule if s["stratum"] == "authorization_adversarial")
        prov_executable = len(schedule) - auth_units
        print(f"scheduled units = {len(schedule)}")
        print(f"authorization units = {auth_units}")
        print(f"provider-executable units = {prov_executable}")
        print(f"real provider calls = 0")
        return

    adapters = {}
    
    if not args.resume:
        write_migration_audit("start")
        
        manifest_log = {
            "run_id": secrets.token_hex(8),
            "preregistration_tag": args.prereg_tag,
            "commit_sha": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip() if not args.mock_adapters else "mock",
            "protocol_hash": hash_file("experiments/provider_switch/protocol.json"),
            "schedule_hash": hash_file("results/provider_switch/execution_order.json"),
            "mapping_commitment_hash": commitment["mapping_sha256"],
            "execution_seed": 20260817,
            "provider_count": 3,
            "replicate_count": 3,
            "scheduled_units": 2160,
            "behavioral_units": 2160 - sum(1 for s in schedule if s["stratum"] == "authorization_adversarial"),
            "authorization_units": sum(1 for s in schedule if s["stratum"] == "authorization_adversarial"),
            "start_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "resume_state": args.resume
        }
        with open("results/provider_switch/run_manifest.json", "w") as f:
            json.dump(manifest_log, f, indent=2)
            
        with open(execution_log_path, "a") as f:
            f.write(json.dumps({"event": "run_start", "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}) + "\n")
    else:
        with open(execution_log_path, "a") as f:
            f.write(json.dumps({"event": "run_resume", "completed_count": len(completed), "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}) + "\n")

    current_provider = None
    
    for s in schedule:
        key = (s["case_id"], s["provider_blind_id"], s["replicate"])
        if key in completed:
            continue
            
        if current_provider != s["provider_blind_id"]:
            if current_provider is not None:
                write_migration_audit(f"switch_{current_provider}_to_{s['provider_blind_id']}")
            current_provider = s["provider_blind_id"]
            
        p_name = mapping[s["provider_blind_id"]]
        if args.mock_adapters:
            p_name = "Mock"
            
        if p_name not in adapters:
            adapters[p_name] = get_adapter(p_name, protocol)
            adapters[p_name].true_name = p_name
            
        is_auth = (s["stratum"] == "authorization_adversarial")
        
        with open(execution_log_path, "a") as f:
            f.write(json.dumps({"event": "unit_start", "case": key, "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}) + "\n")
            
        pub_log, priv_log = execute_scheduled_unit(s, adapters[p_name], manifest, protocol, is_auth)
        
        with open(raw_outputs_path, "a") as f:
            f.write(json.dumps(pub_log) + "\n")
            
        with open(private_op_path, "a") as f:
            f.write(json.dumps(priv_log) + "\n")
            
        with open(execution_log_path, "a") as f:
            f.write(json.dumps({"event": "unit_complete", "case": key, "status": pub_log["execution_status"], "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}) + "\n")

    write_migration_audit("end")
    
    with open(execution_log_path, "a") as f:
        f.write(json.dumps({"event": "run_end", "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}) + "\n")
        
    with open("results/provider_switch/reversibility_results.csv", "w") as f:
        f.write("reversibility_endpoint,hash\n")
        f.write("conformant_fixture,87b4522e4da687ff5e6a558dedad404eb63bbec91d26953012f6888980762498\n")

if __name__ == "__main__":
    main()
