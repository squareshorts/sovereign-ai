import os
import urllib.request
import zipfile
import subprocess
import json

def main():
    print("Downloading JRE...")
    jre_url = "https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.12%2B7/OpenJDK17U-jre_x64_windows_hotspot_17.0.12_7.zip"
    urllib.request.urlretrieve(jre_url, "jre.zip")
    
    print("Extracting JRE...")
    with zipfile.ZipFile("jre.zip", 'r') as zip_ref:
        zip_ref.extractall("jre")
        
    print("Downloading Synthea v3.2.0...")
    synthea_url = "https://github.com/synthetichealth/synthea/releases/download/v3.2.0/synthea-with-dependencies.jar"
    urllib.request.urlretrieve(synthea_url, "synthea.jar")
    
    # Write synthea.properties
    props = """
exporter.fhir.export = true
exporter.fhir_stu3.export = false
exporter.fhir_dstu2.export = false
"""
    with open("synthea.properties", "w") as f:
        f.write(props.strip())

    java_exe = os.path.join(os.getcwd(), "jre", "jdk-17.0.12+7-jre", "bin", "java.exe")
    
    print("Running Synthea...")
    subprocess.run([java_exe, "-jar", "synthea.jar", "-p", "5", "-s", "20260817"], check=True)
    
    print("Synthea run complete. Output files:")
    for root, dirs, files in os.walk("output"):
        for f in files:
            print(f)
            
if __name__ == "__main__":
    main()
