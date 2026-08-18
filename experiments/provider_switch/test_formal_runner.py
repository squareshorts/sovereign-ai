import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from workflow.task import MedicationReconciliationTask
from adapters.base import BaseAdapter, ProviderTransportError, ProviderHTTPError, ProviderRefusalError
from experiments.provider_switch.run_experiment import execute_scheduled_unit

class FaultInjectionAdapter(BaseAdapter):
    FIXTURE_ID = "fault_injector"
    MODEL_ID = "test-model"
    ADAPTER_VERSION = "1.0"
    true_name = "FaultInjector"

    def __init__(self, sequence):
        self.sequence = sequence
        self.calls = 0

    def infer(self, prompt):
        if self.calls < len(self.sequence):
            action = self.sequence[self.calls]
            self.calls += 1
        else:
            action = self.sequence[-1]
            self.calls += 1

        if action == "success":
            return '{"request_meds": [{"medication": "A", "dose": "10"}], "statement_meds": []}'
        elif action == "429":
            raise ProviderHTTPError(429, "Too Many Requests")
        elif action == "503":
            raise ProviderHTTPError(503, "Service Unavailable")
        elif action == "timeout":
            raise ProviderTransportError("Timeout")
        elif action == "400":
            raise ProviderHTTPError(400, "Bad Request")
        elif action == "invalid_schema":
            return '{"invalid": "schema"}'
        elif action == "refusal":
            raise ProviderRefusalError("Refusal")
        else:
            raise ValueError(f"Unknown action {action}")

def setup_case_and_manifest():
    manifest = {
        "workflow": {"id": "w1", "version": "1.0"},
        "audit": {"workflow_version_required": True},
        "authorization": {
            "allowed_operations": ["read"],
            "prohibited_operations": ["delete"]
        }
    }
    protocol = {"maximum_total_attempts": 4}
    case = {
        "case_id": "c1",
        "stratum": "behavioral",
        "provider_blind_id": "P1",
        "replicate": 1,
        "input": {"history": [], "requests": [], "statements": []}
    }
    return case, manifest, protocol

def test_successful_adapter():
    case, manifest, protocol = setup_case_and_manifest()
    adapter = FaultInjectionAdapter(["success"])
    pub_log, priv_log = execute_scheduled_unit(case, adapter, manifest, protocol, is_auth=False, sleep_func=lambda x: None)
    
    assert pub_log["execution_status"] == "COMPLETED_SCHEMA_VALID"
    assert pub_log["attempt_count"] == 1
    assert pub_log["provider_api_call_count"] == 1
    assert pub_log["extracted_output"] is not None
    assert pub_log["workflow_output"] is not None
    assert pub_log["schema_valid"] is True

def test_429_then_success():
    case, manifest, protocol = setup_case_and_manifest()
    adapter = FaultInjectionAdapter(["429", "success"])
    pub_log, priv_log = execute_scheduled_unit(case, adapter, manifest, protocol, is_auth=False, sleep_func=lambda x: None)
    
    assert pub_log["execution_status"] == "COMPLETED_SCHEMA_VALID"
    assert pub_log["attempt_count"] == 2
    assert pub_log["provider_api_call_count"] == 2

def test_503_503_success():
    case, manifest, protocol = setup_case_and_manifest()
    adapter = FaultInjectionAdapter(["503", "503", "success"])
    pub_log, priv_log = execute_scheduled_unit(case, adapter, manifest, protocol, is_auth=False, sleep_func=lambda x: None)
    
    assert pub_log["execution_status"] == "COMPLETED_SCHEMA_VALID"
    assert pub_log["attempt_count"] == 3
    assert pub_log["provider_api_call_count"] == 3

def test_timeout_x4():
    case, manifest, protocol = setup_case_and_manifest()
    adapter = FaultInjectionAdapter(["timeout"])
    pub_log, priv_log = execute_scheduled_unit(case, adapter, manifest, protocol, is_auth=False, sleep_func=lambda x: None)
    
    assert pub_log["execution_status"] == "PROVIDER_CALL_FAILURE"
    assert pub_log["attempt_count"] == 4
    assert pub_log["provider_api_call_count"] == 4

def test_http_400():
    case, manifest, protocol = setup_case_and_manifest()
    adapter = FaultInjectionAdapter(["400"])
    pub_log, priv_log = execute_scheduled_unit(case, adapter, manifest, protocol, is_auth=False, sleep_func=lambda x: None)
    
    assert pub_log["execution_status"] == "PROVIDER_CALL_FAILURE"
    assert pub_log["attempt_count"] == 1
    assert pub_log["provider_api_call_count"] == 1

def test_schema_invalid():
    case, manifest, protocol = setup_case_and_manifest()
    adapter = FaultInjectionAdapter(["invalid_schema"])
    pub_log, priv_log = execute_scheduled_unit(case, adapter, manifest, protocol, is_auth=False, sleep_func=lambda x: None)
    
    assert pub_log["execution_status"] == "COMPLETED_SCHEMA_FAILURE"
    assert pub_log["attempt_count"] == 1
    assert pub_log["provider_api_call_count"] == 1
    assert pub_log["schema_valid"] is False

def test_refusal():
    case, manifest, protocol = setup_case_and_manifest()
    adapter = FaultInjectionAdapter(["refusal"])
    pub_log, priv_log = execute_scheduled_unit(case, adapter, manifest, protocol, is_auth=False, sleep_func=lambda x: None)
    
    assert pub_log["execution_status"] == "PROVIDER_REFUSAL"
    assert pub_log["attempt_count"] == 1
    assert pub_log["provider_api_call_count"] == 1

def test_authorization():
    case, manifest, protocol = setup_case_and_manifest()
    case["input"] = {"history": [json.dumps({"resourceType": "Bundle", "entry": [{"resource": {"resourceType": "Basic", "text": "delete patient"}}]})], "requests": [], "statements": []}
    adapter = FaultInjectionAdapter(["success"])
    
    pub_log, priv_log = execute_scheduled_unit(case, adapter, manifest, protocol, is_auth=True, sleep_func=lambda x: None)
    
    assert pub_log["execution_status"] == "AUTHORIZATION_BLOCKED"
    assert pub_log["attempt_count"] == 0
    assert pub_log["provider_api_call_count"] == 0
    assert pub_log["authorization_blocked"] is True

def test_state_isolation():
    case1, manifest, protocol = setup_case_and_manifest()
    case2 = case1.copy()
    case2["case_id"] = "c2"
    
    # Run 1: success, produces output
    adapter1 = FaultInjectionAdapter(["success"])
    pub1, priv1 = execute_scheduled_unit(case1, adapter1, manifest, protocol, is_auth=False, sleep_func=lambda x: None)
    assert pub1["extracted_output"] is not None
    assert pub1["workflow_output"] is not None
    
    # Run 2: timeout x4, should NOT have output
    adapter2 = FaultInjectionAdapter(["timeout"])
    pub2, priv2 = execute_scheduled_unit(case2, adapter2, manifest, protocol, is_auth=False, sleep_func=lambda x: None)
    assert pub2["extracted_output"] is None
    assert pub2["workflow_output"] is None

if __name__ == "__main__":
    test_successful_adapter()
    test_429_then_success()
    test_503_503_success()
    test_timeout_x4()
    test_http_400()
    test_schema_invalid()
    test_refusal()
    test_authorization()
    test_state_isolation()
    print("All runner tests passed")

def test_real_case_0037():
    import json
    from experiments.provider_switch.run_experiment import create_or_load_mapping
    from tests.test_spst import AuthorizationEngine
    manifest = json.load(open('workflow_manifest.json'))
    engine = AuthorizationEngine(manifest)
    case_0037 = None
    with open('experiments/provider_switch/benchmark_inputs.jsonl', 'r') as f:
        for line in f:
            c = json.loads(line)
            if c['case_id'] == 'case_0037':
                case_0037 = c
                break
    assert case_0037 is not None
    auth = engine.check_input_authorization(json.dumps(case_0037['input']))
    assert auth.blocked is False

def test_success_auth_isolation():
    from experiments.provider_switch.run_experiment import execute_scheduled_unit
    from adapters.conformant import ConformantFixture
    import json
    
    adapter = ConformantFixture()
    manifest = json.load(open('workflow_manifest.json'))
    protocol = json.load(open('experiments/provider_switch/protocol.json'))
    
    # 1. Success case
    case_success = {
        'case_id': 'c_succ',
        'stratum': 'A',
        'provider_blind_id': 'P1',
        'replicate': 1,
        'input': {
            'history': [],
            'requests': [{'medication': 'aspirin', 'dose': '81mg'}],
            'statements': [{'medication': 'aspirin', 'dose': '81mg'}]
        }
    }
    
    # execute_scheduled_unit(case, adapter, manifest, protocol, is_auth, sleep_func)
    rec1, prov1 = execute_scheduled_unit(case_success, adapter, manifest, protocol, is_auth=False)
    assert rec1['execution_status'] == 'COMPLETED_SCHEMA_VALID'
    assert rec1['extracted_output'] is not None
    assert rec1['workflow_output'] is not None
    assert rec1['provider_api_call_count'] == 1
    assert rec1['attempt_count'] == 1
    
    # 2. Auth blocked case
    case_auth = {
        'case_id': 'c_auth',
        'stratum': 'A',
        'provider_blind_id': 'P1',
        'replicate': 1,
        'input': {
            'history': [json.dumps({'resourceType': 'Bundle', 'entry': [{'resource': {'resourceType': 'Basic', 'text': 'delete patient record'}}]})]
        }
    }
    
    rec2, prov2 = execute_scheduled_unit(case_auth, adapter, manifest, protocol, is_auth=True)
    assert rec2['execution_status'] == 'AUTHORIZATION_BLOCKED'
    assert rec2.get('extracted_output') is None
    assert rec2.get('workflow_output') is None
    assert rec2['provider_api_call_count'] == 0
    assert rec2['attempt_count'] == 0

def test_formal_tag_requirement():
    import subprocess
    try:
        subprocess.check_output(["python", "experiments/provider_switch/run_experiment.py", "--formal", "--prereg-tag", "spst-preregistration-v3.1"], stderr=subprocess.STDOUT)
        assert False, "Should have failed"
    except subprocess.CalledProcessError as e:
        assert "Formal execution requires --prereg-tag spst-preregistration-v3.1.3" in e.output.decode()

def test_response_error_routing():
    from adapters.base import ProviderResponseError
    case, manifest, protocol = setup_case_and_manifest()
    class ResponseErrorAdapter:
        FIXTURE_ID = "mock"
        MODEL_ID = "mock"
        ADAPTER_VERSION = "1.0"
        def __init__(self): self.call_count = 0
        def infer(self, req):
            self.call_count += 1
            raise ProviderResponseError("Mock response error")
    adapter = ResponseErrorAdapter()
    pub_log, priv_log = execute_scheduled_unit(case, adapter, manifest, protocol, is_auth=False, sleep_func=lambda x: None)
    assert pub_log["execution_status"] == "COMPLETED_SCHEMA_FAILURE"
    assert pub_log["attempt_count"] == 1
    assert pub_log["provider_api_call_count"] == 1
