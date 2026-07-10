import os
from pathlib import Path
import re

def extract_patient_id(folder_name):
    """Extract patient ID from folder name like '001_SER00001' -> '001'"""
    match = re.match(r'^(\d+)_', folder_name)
    if match:
        return match.group(1)
    return None

def count_patients_in_labels():
    """Count unique patients in Full_data_converted/labels"""
    labels_path = Path(r"c:\Projekt_badawczy\Full_data_converted\labels")
    
    if not labels_path.exists():
        print(f"Error: {labels_path} does not exist")
        return set()
    
    patient_ids = set()
    
    for folder in labels_path.iterdir():
        if folder.is_dir():
            patient_id = extract_patient_id(folder.name)
            if patient_id:
                patient_ids.add(patient_id)
    
    return patient_ids

def count_patients_in_sanna_tumors():
    """Count patients in SANNA_FULL/tumors/Liver3D_labels"""
    tumors_path = Path(r"c:\Projekt_badawczy\SANNA_FULL\tumors\Liver3D_labels")
    
    if not tumors_path.exists():
        print(f"Error: {tumors_path} does not exist")
        return set()
    
    patient_ids = set()
    
    for folder in tumors_path.iterdir():
        if folder.is_dir():
            # These folders are named like '186', '187', etc.
            patient_ids.add(folder.name)
    
    return patient_ids

def main():
    print("="*70)
    print("PATIENT COUNT ANALYSIS")
    print("="*70)
    
    # Count patients in Full_data_converted/labels
    print("\n1. Counting patients in Full_data_converted/labels...")
    labels_patients = count_patients_in_labels()
    print(f"   Found {len(labels_patients)} unique patients")
    print(f"   Patient IDs: {sorted(labels_patients)}")
    
    # Count patients in SANNA_FULL/tumors
    print("\n2. Counting patients in SANNA_FULL/tumors/Liver3D_labels...")
    tumor_patients = count_patients_in_sanna_tumors()
    print(f"   Found {len(tumor_patients)} unique patients")
    print(f"   Patient IDs: {sorted(tumor_patients)}")
    
    # Note: Looking for XXX marks
    print("\n3. Checking for XXX marks...")
    xxx_marked = set()
    tumors_path = Path(r"c:\Projekt_badawczy\SANNA_FULL\tumors\Liver3D_labels")
    
    for patient_folder in tumors_path.iterdir():
        if patient_folder.is_dir():
            # Check for XXX in folder name
            if 'XXX' in patient_folder.name or 'xxx' in patient_folder.name:
                xxx_marked.add(patient_folder.name)
            
            # Check for XXX in any files inside
            for file in patient_folder.rglob('*'):
                if 'XXX' in file.name or 'xxx' in file.name:
                    xxx_marked.add(patient_folder.name)
                    break
    
    if xxx_marked:
        print(f"   Found {len(xxx_marked)} patients with XXX marks: {sorted(xxx_marked)}")
    else:
        print("   No XXX marks found in patient folders or files")
    
    # Patients without XXX marks
    tumor_patients_no_xxx = tumor_patients - xxx_marked
    print(f"\n4. Tumor patients WITHOUT XXX marks: {len(tumor_patients_no_xxx)}")
    print(f"   Patient IDs: {sorted(tumor_patients_no_xxx)}")
    
    # Combined unique patients
    print("\n5. COMBINED ANALYSIS:")
    combined_patients = labels_patients.union(tumor_patients_no_xxx)
    print(f"   Total unique patients (labels + tumors without XXX): {len(combined_patients)}")
    
    # Breakdown
    only_in_labels = labels_patients - tumor_patients_no_xxx
    only_in_tumors = tumor_patients_no_xxx - labels_patients
    in_both = labels_patients.intersection(tumor_patients_no_xxx)
    
    print(f"\n   - Only in Full_data_converted/labels: {len(only_in_labels)}")
    print(f"     IDs: {sorted(only_in_labels)}")
    print(f"\n   - Only in SANNA_FULL/tumors (no XXX): {len(only_in_tumors)}")
    print(f"     IDs: {sorted(only_in_tumors)}")
    print(f"\n   - In both: {len(in_both)}")
    print(f"     IDs: {sorted(in_both)}")
    
    print("\n" + "="*70)
    print(f"FINAL COUNT: {len(combined_patients)} unique patients total")
    print("="*70)
    
    # Save to file
    output_file = Path(r"c:\Projekt_badawczy\unique_patients_count.txt")
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("PATIENT COUNT ANALYSIS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Full_data_converted/labels: {len(labels_patients)} patients\n")
        f.write(f"SANNA_FULL/tumors (no XXX): {len(tumor_patients_no_xxx)} patients\n")
        f.write(f"Combined unique: {len(combined_patients)} patients\n\n")
        f.write("All unique patient IDs:\n")
        f.write(", ".join(sorted(combined_patients)))
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
