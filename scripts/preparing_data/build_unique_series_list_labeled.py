import argparse
from pathlib import Path

import pandas as pd


def collect_labeled_patients(label_roots: list[Path]) -> set[str]:
    labeled = set()
    for root in label_roots:
        if not root.exists():
            continue
        for path in root.glob("**/*.nii*"):
            name = path.stem
            if name.endswith(".nii"):
                name = Path(name).stem
            base_name = name
            if base_name.startswith("Untitled"):
                base_name = base_name[len("Untitled"):].lstrip("-_ ")
            parent_token = path.parent.name
            if parent_token.isdigit():
                labeled.add(parent_token.zfill(3))
                continue
            patient = base_name.split("_")[0]
            if patient.isdigit():
                labeled.add(patient.zfill(3))
    return labeled


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter unique_series_list to labeled patients only")
    parser.add_argument(
        "--input",
        type=str,
        default="C:/Projekt_badawczy/Full_data_converted/unique_series_list.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="C:/Projekt_badawczy/Full_data_converted/unique_series_list_labeled.csv",
    )
    parser.add_argument(
        "--label-roots",
        type=str,
        default=(
            "C:/Projekt_badawczy/SANNA_FULL/Liver3D_labels;"
            "C:/Projekt_badawczy/SANNA_FULL/tumors/Liver3D_labels"
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    label_roots = [Path(p) for p in args.label_roots.split(";") if p]
    labeled_patients = collect_labeled_patients(label_roots)

    df = pd.read_csv(input_path, low_memory=False)
    if "patient_id" not in df.columns:
        raise ValueError("CSV must contain patient_id column")

    df["patient_id"] = df["patient_id"].apply(lambda x: str(int(x)).zfill(3))
    filtered = df[df["patient_id"].isin(labeled_patients)]
    filtered.to_csv(output_path, index=False)

    print(f"Labeled patients: {len(labeled_patients)}")
    print(f"Rows written: {len(filtered)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
