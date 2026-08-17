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
    mapping_path = "experiments/provider_switch/private/provider_mapping_private.json"
    commitment_path = "experiments/provider_switch/provider_mapping_commitment.json"
    
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
    # 3 replicates
    schedule = []
    base_order = ["P1", "P2", "P3"]
    for rep in range(1, 4):
        rng = random.Random(20260817 + rep)
        rep_cases = list(cases)
        rng.shuffle(rep_cases)
        for j, case in enumerate(rep_cases):
            rotation = (j + rep - 1) % 3
            # Rotate base_order
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

def is_retryable(error_msg):
    codes = ["429", "500", "502", "503", "504", "timeout", "temporary"]
    return any(c in str(error_msg).lower() for c in codes)

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
        # For testing
        class MockAdapter:
            FIXTURE_ID = "Mock"
            MODEL_ID = "mock-model"
            ADAPTER_VERSION = "1.0"
            def infer(self, req):
                return '{"request_meds": [], "statement_meds": []}'
        return MockAdapter()

def check_preconditions():
    # HEAD check
    head_rev = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    tag_rev = subprocess.check_output(["git", "rev-parse", "spst-preregistration-v2^{}"]).decode().strip()
    if head_rev != tag_rev:
        raise ValueError("HEAD is not at spst-preregistration-v2")
    
    # Clean tree
    status = subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()
    if status:
        raise ValueError("Working tree is not clean")
        
    # Protected hashes check (if artifact_hashes.json exists)
    if os.path.exists("artifact_hashes.json"):
        with open("artifact_hashes.json", "r") as f:
            expected = json.load(f)
        for path, expected_hash in expected.items():
            if os.path.exists(path):
                if hash_file(path) != expected_hash:
                    raise ValueError(f"Protected hash mismatch for {path}")
                    
    # Credentials
    for k in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]:
        if not os.environ.get(k):
            raise ValueError(f"Missing credential {k}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--mock-adapters", action="store_true", help="Use mock adapters for testing")
    parser.add_argument("--prereg-tag", type=str)
    args = parser.parse_args()

    if not args.dry_run and not args.formal:
        print("Must specify --dry-run or --formal")
        sys.exit(1)
        
    if args.formal and args.prereg_tag != "spst-preregistration-v2":
        if not args.mock_adapters: # Only skip tag check if testing
            raise ValueError("Formal execution requires --prereg-tag spst-preregistration-v2")

    if args.formal and not args.mock_adapters:
        check_preconditions()

    with open("experiments/provider_switch/protocol.json", "r") as f:
        protocol = json.load(f)
        
    with open("workflow_manifest.json", "r") as f:
        manifest = json.load(f)

    mapping = create_or_load_mapping()
    
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
    
    if os.path.exists(raw_outputs_path):
        with open(raw_outputs_path, "r") as f:
            for line in f:
                item = json.loads(line)
                completed.add((item["case_id"], item["provider_blind_id"], item["replicate"]))
                
    if args.dry_run:
        print(f"Dry run constructed schedule of {len(schedule)} units.")
        return

    from workflow.task import MedicationReconciliationTask
    adapters = {}
    
    for s in schedule:
        key = (s["case_id"], s["provider_blind_id"], s["replicate"])
        if key in completed:
            continue
            
        p_name = mapping[s["provider_blind_id"]]
        if args.mock_adapters:
            p_name = "Mock"
            
        if p_name not in adapters:
            adapters[p_name] = get_adapter(p_name, protocol)
            
        task = MedicationReconciliationTask(manifest, adapters[p_name])
        
        is_auth = (s["stratum"] == "authorization_adversarial")
        
        attempts = 0
        success = False
        final_status = ""
        output_data = None
        prov = None
        auth = None
        
        while attempts < protocol["maximum_total_attempts"] and not success:
            attempts += 1
            if is_auth:
                output_data, prov, auth, schema_ok, errs = task.execute(s["case_id"], s["input"])
                final_status = "AUTHORIZATION_BLOCKED"
                success = True
                break
                
            try:
                output_data, prov, auth, schema_ok, errs = task.execute(s["case_id"], s["input"])
                if schema_ok:
                    final_status = "COMPLETED_SCHEMA_VALID"
                else:
                    if prov.execution_status == "FAILED_PARSE" or prov.execution_status == "FAILED_EXTRACTION_FIELDS" or prov.execution_status == "FAILED_OUTPUT_SCHEMA":
                        final_status = "COMPLETED_SCHEMA_FAILURE"
                    else:
                        final_status = prov.execution_status
                
                # Provider Refusal logic can be added if models return exact refusal shapes,
                # but currently tracked as schema failure if it doesn't parse to JSON.
                
                success = True
            except Exception as e:
                err_str = str(e)
                if is_retryable(err_str) and attempts < protocol["maximum_total_attempts"]:
                    time.sleep(2 ** (attempts - 1))
                else:
                    final_status = "TRANSPORT_FAILURE_AFTER_RETRIES"
                    prov = task.execute(s["case_id"], s["input"])[1] # Get base prov
                    prov.execution_status = final_status
                    success = True # Give up and log
                    
        # Log public raw output (blinded)
        pub_log = {
            "case_id": s["case_id"],
            "stratum": s["stratum"],
            "provider_blind_id": s["provider_blind_id"],
            "replicate": s["replicate"],
            "execution_status": final_status,
            "schema_valid": final_status == "COMPLETED_SCHEMA_VALID",
            "output": output_data
        }
        with open(raw_outputs_path, "a") as f:
            f.write(json.dumps(pub_log) + "\n")
            
        # Log private operational
        priv_log = {
            "case_id": s["case_id"],
            "stratum": s["stratum"],
            "provider_blind_id": s["provider_blind_id"],
            "true_provider": p_name,
            "replicate": s["replicate"],
            "execution_status": final_status,
            "provider_api_call_count": 0 if is_auth else attempts,
            "schema_valid": final_status == "COMPLETED_SCHEMA_VALID",
            "output": output_data,
            "provenance": prov.to_dict() if prov else {},
            "auth": auth.to_dict() if auth else {}
        }
        with open(private_op_path, "a") as f:
            f.write(json.dumps(priv_log) + "\n")

if __name__ == "__main__":
    main()
