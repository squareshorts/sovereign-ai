import json
import os
import csv
import hashlib
import sys

def hash_data(data):
    serialized = json.dumps(data, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

def hash_file(filepath):
    if not os.path.exists(filepath): return None
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("Starting unblinding...")
    
    # 1. Verify blinded_output_hashes.json
    hash_path = "results/provider_switch/blinded_output_hashes.json"
    if not os.path.exists(hash_path):
        raise FileNotFoundError(f"Missing {hash_path}")
        
    with open(hash_path, "r") as f:
        expected_hashes = json.load(f)
        
    for path, exp_h in expected_hashes.items():
        if exp_h is None:
            continue
        if hash_file(path) != exp_h:
            raise ValueError(f"Hash mismatch for {path}! Unblinding aborted.")
            
    print("Primary output hashes verified.")
    
    # 2. Verify provider mapping commitment
    mapping_path = "experiments/provider_switch/private/provider_mapping_private.json"
    commitment_path = "experiments/provider_switch/provider_mapping_commitment.json"
    
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Missing {mapping_path}")
    if not os.path.exists(commitment_path):
        raise FileNotFoundError(f"Missing {commitment_path}")
        
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
        
    with open(commitment_path, "r") as f:
        commitment = json.load(f)
        
    if commitment["mapping_sha256"] != hash_data(mapping):
        raise ValueError("Provider mapping hash does not match commitment!")
        
    print("Provider mapping commitment verified.")
    
    # 3. Generate identity-labeled copies
    in_csv = "results/provider_switch/replicate_summary.csv"
    out_csv = "results/provider_switch/replicate_summary_unblinded.csv"
    
    if os.path.exists(in_csv):
        with open(in_csv, "r") as fin, open(out_csv, "w", newline='') as fout:
            reader = csv.reader(fin)
            writer = csv.writer(fout)
            
            headers = next(reader)
            if headers[0] == "provider":
                headers[0] = "true_provider"
                
            writer.writerow(headers)
            for row in reader:
                blind_id = row[0]
                row[0] = mapping.get(blind_id, blind_id)
                writer.writerow(row)
                
    print(f"Unblinded results written to {out_csv}")

if __name__ == "__main__":
    main()
