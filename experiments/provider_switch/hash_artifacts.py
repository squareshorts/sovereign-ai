import os
import json
import hashlib

def hash_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    protocol_path = os.path.join(os.path.dirname(__file__), "protocol.json")
    with open(protocol_path, "r") as f:
        protocol = json.load(f)
        
    protected_files = protocol["protected_files"]
    
    hashes = {}
    for pf in protected_files:
        full_path = os.path.join(os.path.dirname(__file__), "..", "..", pf)
        if os.path.exists(full_path):
            hashes[pf] = hash_file(full_path)
        else:
            print(f"Warning: Protected file {pf} not found.")
            
    with open(os.path.join(os.path.dirname(__file__), "..", "..", "artifact_hashes.json"), "w") as f:
        json.dump(hashes, f, indent=2)
        
    print(f"Hashed {len(hashes)} protected artifacts.")

if __name__ == "__main__":
    main()
