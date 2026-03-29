@echo off
REM Example wrapper for full data preparation pipeline.

call ..\..\.venv\Scripts\activate.bat

python run_full_pipeline.py ^
  --base-dicom-root "C:\Projekt_badawczy\SANNA_FULL\Liver3D_originals" ^
  --base-nifti-root "C:\Projekt_badawczy\SANNA_FULL\Liver3D_labels" ^
  --tumor-dicom-root "C:\Projekt_badawczy\SANNA_FULL\tumors\Liver3D_originals" ^
  --tumor-nifti-root "C:\Projekt_badawczy\SANNA_FULL\tumors\Liver3D_labels" ^
  --output-dir "C:\Projekt_badawczy\Full_data_converted" ^
  --n-folds 6 ^
  --verbose
