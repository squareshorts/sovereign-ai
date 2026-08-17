import json
import os

def main():
    with open("experiments/provider_switch/protocol.json", "r") as f:
        protocol = json.load(f)
        
    with open("experiments/provider_switch/PROTOCOL.md", "r", encoding="utf-8") as f:
        md = f.read()
        
    # Check consistency
    assert str(protocol["formal_cases"]) in md
    assert str(protocol["behavioral_cases"]) in md
    assert str(protocol["authorization_cases"]) in md
    assert str(protocol["providers"]) in md
    assert str(protocol["replicates"]) in md
    assert str(protocol["execution_seed"]) in md
    assert protocol["synthea_repository"] in md
    assert protocol["synthea_version_sha"] in md
    
    print("test_protocol_consistency: ALL PASSED")

if __name__ == "__main__":
    main()
