import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report patients with more than N series in unique_series_list.csv"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="C:/Projekt_badawczy/Full_data_converted/unique_series_list.csv",
    )
    parser.add_argument("--max-series", type=int, default=4)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    if "patient_id" not in df.columns or "series_id" not in df.columns:
        raise ValueError("CSV must contain patient_id and series_id columns")

    counts = df.groupby("patient_id")["series_id"].nunique().sort_values(ascending=False)
    over = counts[counts > args.max_series]

    if over.empty:
        print("No patients exceed the series limit.")
        return

    print(f"Patients with more than {args.max_series} series:")
    for patient_id, count in over.items():
        pid = str(int(patient_id)).zfill(3)
        print(f"{pid}: {count}")


if __name__ == "__main__":
    main()
