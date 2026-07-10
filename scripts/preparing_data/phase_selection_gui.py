import argparse
import io
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
import pydicom
import tkinter as tk
from tkinter import ttk, messagebox


DICOM_ROOTS_DEFAULT = (
    "C:/Projekt_badawczy/SANNA_FULL/Liver3D_originals;"
    "C:/Projekt_badawczy/SANNA_FULL/tumors/Liver3D_originals"
)
CSV_DEFAULT = "C:\\Projekt_badawczy\\tomography_segmentation\\generated_helper_csv_files\\unique_series_list_labeled.csv"
OUTPUT_DEFAULT = "C:\\Projekt_badawczy\\tomography_segmentation\\generated_helper_csv_files\\phase_mapping_manual.csv"
LABEL_ROOTS_DEFAULT = (
    "C:/Projekt_badawczy/SANNA_FULL/Liver3D_labels;"
    "C:/Projekt_badawczy/SANNA_FULL/tumors/Liver3D_labels"
)

PHASES = [
    "unknown",
    "precontrast",
    "arterial",
    "venous",
    "delayed",
    "hepatic",
]

PHASE_PATTERNS = {
    "precontrast": [r"\bpre\b", r"\bnative\b", r"\bnon-contrast\b", r"\bplain\b", r"\bwithout\b"],
    "arterial": [r"\barterial\b", r"\bart\b", r"\bearly\b", r"\bhap\b"],
    "venous": [r"\bvenous\b", r"\bportal\b", r"\bpv\b", r"\bpvp\b", r"\bwrotna\b"],
    "delayed": [r"\bdelayed\b", r"\blate\b", r"\bequilibrium\b"],
    "hepatic": [r"\bhepatic\b", r"\bliver\b"],
}


@dataclass
class SeriesInfo:
    series_id: int
    series_dir: Path
    slices: int
    spacing: Tuple[float, float, float]
    series_desc: str
    protocol: str
    study: str
    phase_guess: str


def detect_phase(text: str) -> str:
    if not text:
        return "unknown"
    text_lower = text.lower()
    for phase, patterns in PHASE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return phase
    return "unknown"


def normalize_label_name(name: str) -> str:
    if name.startswith("Untitled"):
        return name[len("Untitled"):].lstrip("-_ ")
    return name


def parse_patient_id_from_label_name(name: str) -> Optional[str]:
    name = normalize_label_name(name)
    match = re.match(r"^(\d+)", name)
    if not match:
        return None
    return match.group(1)


def parse_series_id_from_label_name(name: str) -> Optional[int]:
    name = normalize_label_name(name)
    match = re.search(r"_(\d{2})", name)
    if not match:
        return None
    return int(match.group(1))


def load_first_slice(series_dir: Path) -> Optional[np.ndarray]:
    dicom_files = sorted([p for p in series_dir.glob("*") if p.is_file()])
    if not dicom_files:
        return None
    ds0 = None
    for path in dicom_files:
        try:
            ds0 = pydicom.dcmread(path, force=True)
            break
        except Exception:
            continue
    if ds0 is None:
        return None
    try:
        pixel_array = ds0.pixel_array.astype(np.float32)
    except Exception:
        return None
    slope = float(getattr(ds0, "RescaleSlope", 1.0))
    intercept = float(getattr(ds0, "RescaleIntercept", -1024.0))
    hu = pixel_array * slope + intercept
    return hu


def series_metadata(series_dir: Path) -> Optional[SeriesInfo]:
    dicom_files = sorted([p for p in series_dir.glob("*") if p.is_file()])
    if not dicom_files:
        return None
    ds0 = None
    for path in dicom_files:
        try:
            ds0 = pydicom.dcmread(path, force=True, stop_before_pixels=True)
            break
        except Exception:
            continue
    if ds0 is None:
        return None
    spacing_y, spacing_x = map(float, getattr(ds0, "PixelSpacing", [1.0, 1.0]))
    spacing_z = float(getattr(ds0, "SliceThickness", 1.0))
    series_desc = str(getattr(ds0, "SeriesDescription", "") or "")
    protocol = str(getattr(ds0, "ProtocolName", "") or "")
    study = str(getattr(ds0, "StudyDescription", "") or "")
    phase_guess = detect_phase(" ".join([series_desc, protocol, study]))
    match = re.search(r"SER(\d+)", series_dir.name)
    if not match:
        return None
    series_id = int(match.group(1))
    return SeriesInfo(
        series_id=series_id,
        series_dir=series_dir,
        slices=len(dicom_files),
        spacing=(spacing_x, spacing_y, spacing_z),
        series_desc=series_desc,
        protocol=protocol,
        study=study,
        phase_guess=phase_guess,
    )


def collect_series_dirs(dicom_roots: List[Path], patient_filter: set) -> Dict[str, Dict[int, Path]]:
    series_map: Dict[str, Dict[int, Path]] = {}
    for root in dicom_roots:
        if not root.exists():
            continue
        for patient_dir in root.iterdir():
            if not patient_dir.is_dir():
                continue
            patient_id = patient_dir.name
            if patient_id not in patient_filter:
                continue
            dicoms_dir = patient_dir / "DICOMS"
            if not dicoms_dir.exists():
                continue
            for study_dir in dicoms_dir.iterdir():
                if not study_dir.is_dir():
                    continue
                for series_dir in study_dir.iterdir():
                    if not series_dir.is_dir():
                        continue
                    match = re.search(r"SER(\d+)", series_dir.name)
                    if not match:
                        continue
                    series_id = int(match.group(1))
                    series_map.setdefault(patient_id, {})[series_id] = series_dir
    return series_map


def hu_to_ppm(hu: np.ndarray, window_center: float = 50.0, window_width: float = 350.0) -> bytes:
    lo = window_center - window_width / 2.0
    hi = window_center + window_width / 2.0
    img = np.clip((hu - lo) / (hi - lo), 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)
    h, w = img.shape
    header = f"P5 {w} {h} 255\n".encode("ascii")
    return header + img.tobytes()


def collect_label_info(label_roots: List[Path]) -> Dict[Tuple[str, int], List[Tuple[int, str]]]:
    label_map: Dict[Tuple[str, int], List[Tuple[int, str]]] = {}
    for root in label_roots:
        if not root.exists():
            continue
        for path in root.glob("**/*.nii*"):
            name = path.stem
            if name.endswith(".nii"):
                name = Path(name).stem
            patient_id = parse_patient_id_from_label_name(name)
            series_id = parse_series_id_from_label_name(name)
            if not patient_id or series_id is None:
                continue
            try:
                img = nib.load(str(path))
                shape = img.shape
            except Exception:
                continue
            if len(shape) < 3:
                continue
            z_slices = int(shape[2])
            label_map.setdefault((patient_id.zfill(3), series_id), []).append(
                (z_slices, str(path))
            )
    return label_map


class PhaseSelectorApp:
    def __init__(
        self,
        master: tk.Tk,
        csv_path: Path,
        dicom_roots: List[Path],
        label_roots: List[Path],
        output_csv: Path,
    ) -> None:
        self.master = master
        self.csv_path = csv_path
        self.dicom_roots = dicom_roots
        self.label_roots = label_roots
        self.output_csv = output_csv

        self.df = pd.read_csv(self.csv_path, low_memory=False)
        if "patient_id" not in self.df.columns or "series_id" not in self.df.columns:
            raise ValueError("CSV must contain patient_id and series_id columns")

        self.patient_ids = sorted([str(int(pid)).zfill(3) for pid in self.df["patient_id"].unique()])
        self.allowed_series = {
            str(int(pid)).zfill(3): set(group["series_id"].astype(int))
            for pid, group in self.df.groupby("patient_id")
        }

        self.series_map = collect_series_dirs(self.dicom_roots, set(self.patient_ids))
        self.label_map = collect_label_info(self.label_roots)
        self.series_cache: Dict[str, List[SeriesInfo]] = {}
        self.phase_map: Dict[Tuple[str, int], str] = {}
        self.load_existing_output()

        self.build_ui()

    def build_ui(self) -> None:
        self.master.title("Phase Selection")
        self.master.geometry("2000x900")

        left = ttk.Frame(self.master)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        right = ttk.Frame(self.master)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        ttk.Label(left, text="Patients").pack(anchor=tk.W)
        self.patient_list = tk.Listbox(left, height=30)
        self.patient_list.pack(fill=tk.Y, expand=True)
        for pid in self.patient_ids:
            self.patient_list.insert(tk.END, pid)
        self.patient_list.bind("<<ListboxSelect>>", self.on_patient_select)

        self.tree = ttk.Treeview(
            right,
            columns=(
                "series",
                "slices",
                "label_slices",
                "label_files",
                "label_dir",
                "spacing",
                "desc",
                "protocol",
                "study",
                "guess",
                "phase",
            ),
            show="headings",
        )
        self.tree.heading("series", text="Series")
        self.tree.heading("slices", text="Slices")
        self.tree.heading("label_slices", text="Label Z")
        self.tree.heading("label_files", text="Label Files")
        self.tree.heading("label_dir", text="Label Dir")
        self.tree.heading("spacing", text="Spacing")
        self.tree.heading("desc", text="Series Description")
        self.tree.heading("protocol", text="Protocol")
        self.tree.heading("study", text="Study")
        self.tree.heading("guess", text="Auto")
        self.tree.heading("phase", text="Chosen")
        self.tree.column("series", width=80)
        self.tree.column("slices", width=60)
        self.tree.column("label_slices", width=80)
        self.tree.column("label_files", width=90)
        self.tree.column("label_dir", width=260)
        self.tree.column("spacing", width=120)
        self.tree.column("guess", width=80)
        self.tree.column("phase", width=80)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_series_select)

        controls = ttk.Frame(right)
        controls.pack(fill=tk.X, pady=6)

        ttk.Label(controls, text="Phase:").pack(side=tk.LEFT)
        self.phase_var = tk.StringVar(value=PHASES[0])
        self.phase_combo = ttk.Combobox(controls, textvariable=self.phase_var, values=PHASES, width=15)
        self.phase_combo.pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Set for Selected", command=self.set_phase_for_selected).pack(side=tk.LEFT)
        ttk.Button(controls, text="Save CSV", command=self.save_csv).pack(side=tk.LEFT, padx=8)

        status_frame = ttk.Frame(right)
        status_frame.pack(fill=tk.X, pady=4)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var).pack(anchor=tk.W)

        image_frame = ttk.LabelFrame(right, text="Preview (first slice)")
        image_frame.pack(fill=tk.BOTH, expand=False, pady=6)
        self.image_label = ttk.Label(image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)

    def on_patient_select(self, _event: tk.Event) -> None:
        selection = self.patient_list.curselection()
        if not selection:
            return
        patient_id = self.patient_list.get(selection[0])
        self.load_patient_async(patient_id)

    def load_existing_output(self) -> None:
        if not self.output_csv.exists():
            return
        try:
            existing = pd.read_csv(self.output_csv, low_memory=False)
        except Exception:
            return
        if not {"patient_id", "series_id", "phase"}.issubset(existing.columns):
            return
        for _, row in existing.iterrows():
            patient_id = str(int(row["patient_id"])).zfill(3)
            series_id = int(row["series_id"])
            phase = str(row["phase"])
            self.phase_map[(patient_id, series_id)] = phase

    def load_patient_async(self, patient_id: str) -> None:
        self.status_var.set(f"Loading patient {patient_id}...")
        threading.Thread(target=self.load_patient, args=(patient_id,), daemon=True).start()

    def load_patient(self, patient_id: str) -> None:
        for row in self.tree.get_children():
            self.tree.delete(row)

        if patient_id not in self.series_cache:
            series_dirs = self.series_map.get(patient_id, {})
            allowed = self.allowed_series.get(patient_id, set())
            infos: List[SeriesInfo] = []
            for series_id, series_dir in sorted(series_dirs.items()):
                if series_id not in allowed:
                    continue
                info = series_metadata(series_dir)
                if info:
                    infos.append(info)
            self.series_cache[patient_id] = infos

        def populate() -> None:
            for info in self.series_cache[patient_id]:
                chosen = self.phase_map.get((patient_id, info.series_id), "")
                spacing = f"{info.spacing[0]:.2f},{info.spacing[1]:.2f},{info.spacing[2]:.2f}"
                label_key = (patient_id, info.series_id)
                label_entries = self.label_map.get(label_key, [])
                label_slices = str(label_entries[0][0]) if label_entries else ""
                label_files = str(len(label_entries)) if label_entries else ""
                label_dir = str(Path(label_entries[0][1]).parent) if label_entries else ""
                self.tree.insert(
                    "",
                    tk.END,
                    iid=f"{patient_id}_{info.series_id}",
                    values=(
                        info.series_id,
                        info.slices,
                        label_slices,
                        label_files,
                        label_dir,
                        spacing,
                        info.series_desc,
                        info.protocol,
                        info.study,
                        info.phase_guess,
                        chosen,
                    ),
                )
            self.status_var.set(f"Loaded patient {patient_id}")

        self.master.after(0, populate)

    def on_series_select(self, _event: tk.Event) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        item_id = selected[0]
        patient_id, series_id_str = item_id.split("_")
        series_id = int(series_id_str)

        series_dir = self.series_map.get(patient_id, {}).get(series_id)
        if not series_dir:
            return
        hu = load_first_slice(series_dir)
        if hu is None:
            return

        data = hu_to_ppm(hu)
        self.photo = tk.PhotoImage(data=data)
        self.image_label.configure(image=self.photo)

    def set_phase_for_selected(self) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        phase = self.phase_var.get()
        for item_id in selected:
            patient_id, series_id_str = item_id.split("_")
            series_id = int(series_id_str)
            self.phase_map[(patient_id, series_id)] = phase
            values = list(self.tree.item(item_id, "values"))
            values[-1] = phase
            self.tree.item(item_id, values=values)

    def save_csv(self) -> None:
        if not self.phase_map:
            messagebox.showwarning("No data", "No phase assignments to save.")
            return
        existing_rows = []
        if self.output_csv.exists():
            try:
                existing = pd.read_csv(self.output_csv, low_memory=False)
                if {"patient_id", "series_id", "phase"}.issubset(existing.columns):
                    existing_rows = existing.to_dict("records")
            except Exception:
                existing_rows = []

        existing_map = {
            (str(int(r["patient_id"])).zfill(3), int(r["series_id"])): str(r["phase"]) for r in existing_rows
        }
        existing_map.update(self.phase_map)

        rows = [
            {"patient_id": pid, "series_id": sid, "phase": phase}
            for (pid, sid), phase in sorted(existing_map.items())
        ]
        out_df = pd.DataFrame(rows)
        out_df.to_csv(self.output_csv, index=False)
        messagebox.showinfo("Saved", f"Saved (merged) to {self.output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GUI to assign phases for patients in unique_series_list.csv")
    parser.add_argument("--csv", type=str, default=CSV_DEFAULT)
    parser.add_argument("--dicom-roots", type=str, default=DICOM_ROOTS_DEFAULT)
    parser.add_argument("--label-roots", type=str, default=LABEL_ROOTS_DEFAULT)
    parser.add_argument("--output", type=str, default=OUTPUT_DEFAULT)
    args = parser.parse_args()

    dicom_roots = [Path(p) for p in args.dicom_roots.split(";") if p]
    label_roots = [Path(p) for p in args.label_roots.split(";") if p]
    root = tk.Tk()
    app = PhaseSelectorApp(root, Path(args.csv), dicom_roots, label_roots, Path(args.output))
    root.mainloop()


if __name__ == "__main__":
    main()
