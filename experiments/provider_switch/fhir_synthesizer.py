import random
import uuid
import json
import hashlib
from datetime import datetime, timedelta

def get_seed_rng(seed):
    return random.Random(seed)

MEDICATIONS = [
    ("Acetaminophen 500mg", "500 mg"),
    ("Lisinopril 10mg", "10 mg"),
    ("Atorvastatin 20mg", "20 mg"),
    ("Metformin 1000mg", "1000 mg"),
    ("Amlodipine 5mg", "5 mg"),
    ("Levothyroxine 50mcg", "50 mcg"),
    ("Omeprazole 20mg", "20 mg"),
    ("Simvastatin 40mg", "40 mg"),
    ("Losartan 50mg", "50 mg"),
    ("Albuterol 90mcg", "90 mcg")
]

def make_fhir_bundle(patient_id, meds):
    """Creates an authentic-looking FHIR R4 Bundle containing MedicationRequest and MedicationStatement resources."""
    bundle = {
        "resourceType": "Bundle",
        "id": str(uuid.uuid4()),
        "type": "collection",
        "entry": []
    }
    
    # Add Patient resource
    bundle["entry"].append({
        "fullUrl": f"urn:uuid:{patient_id}",
        "resource": {
            "resourceType": "Patient",
            "id": patient_id,
            "gender": "unknown"
        }
    })
    
    for med_name, dose in meds:
        # Add MedicationRequest
        bundle["entry"].append({
            "fullUrl": f"urn:uuid:{uuid.uuid4()}",
            "resource": {
                "resourceType": "MedicationRequest",
                "status": "active",
                "intent": "order",
                "medicationCodeableConcept": {
                    "text": med_name
                },
                "subject": {
                    "reference": f"urn:uuid:{patient_id}"
                },
                "dosageInstruction": [{
                    "text": dose
                }]
            }
        })
        
        # Add MedicationStatement
        bundle["entry"].append({
            "fullUrl": f"urn:uuid:{uuid.uuid4()}",
            "resource": {
                "resourceType": "MedicationStatement",
                "status": "active",
                "medicationCodeableConcept": {
                    "text": med_name
                },
                "subject": {
                    "reference": f"urn:uuid:{patient_id}"
                },
                "dosage": [{
                    "text": dose
                }]
            }
        })
        
    return bundle

def generate_source_corpus(seed, num_patients):
    """Generates the base FHIR corpus."""
    rng = get_seed_rng(seed)
    corpus = []
    
    for i in range(num_patients):
        patient_id = str(uuid.UUID(int=rng.getrandbits(128), version=4))
        num_meds = rng.randint(3, 8)
        patient_meds = rng.sample(MEDICATIONS, k=num_meds)
        
        bundle = make_fhir_bundle(patient_id, patient_meds)
        
        serialized = json.dumps(bundle, sort_keys=True)
        bundle_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        
        corpus.append({
            "patient_id": patient_id,
            "bundle": bundle,
            "bundle_hash": bundle_hash,
            "original_meds": patient_meds
        })
        
    return corpus
