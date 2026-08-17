import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import json
import subprocess

def main():
    # Make sure we're in the right directory
    assert os.path.exists("experiments/provider_switch/run_experiment.py")
    
    # Run dry-run
    out = subprocess.check_output(["python", "experiments/provider_switch/run_experiment.py", "--dry-run"], text=True)
    assert "2160 units" in out
    
    # Run with mock adapters (this bypasses formal tag check)
    out = subprocess.check_output(["python", "experiments/provider_switch/run_experiment.py", "--formal", "--mock-adapters"], text=True)
    
    with open("results/provider_switch/raw_outputs.jsonl", "r") as f:
        lines = f.readlines()
        assert len(lines) == 2160
        
    auth_blocked = 0
    valid = 0
    for line in lines:
        item = json.loads(line)
        if item["execution_status"] == "AUTHORIZATION_BLOCKED":
            auth_blocked += 1
        elif item["execution_status"] == "COMPLETED_SCHEMA_VALID":
            valid += 1
            
    assert auth_blocked == 270 # 30 cases * 3 replicates * 3 providers
    assert valid == 1890 # 210 cases * 3 replicates * 3 providers
    
    # Check execution_order
    with open("results/provider_switch/execution_order.json", "r") as f:
        order = json.load(f)
        assert len(order) == 2160
        
    print("test_formal_runner: ALL PASSED")
    
if __name__ == "__main__":
    main()
