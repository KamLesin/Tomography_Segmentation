@echo off
REM Training script for liver segmentation

REM Activate virtual environment
call ..\.venv\Scripts\activate.bat

set DATA_ROOT=%~dp0..\Full_data_converted
if not "%SEG_DATA_ROOT%"=="" set DATA_ROOT=%SEG_DATA_ROOT%

set METADATA=%DATA_ROOT%\metadata.csv
set FOLDS=%DATA_ROOT%\cv_folds.csv

REM Run training
python src\train.py ^
    --metadata "%METADATA%" ^
    --folds-csv "%FOLDS%" ^
    --data-root "%DATA_ROOT%" ^
    --fold 0 ^
    --epochs 15 ^
    --batch-size 4 ^
    --lr 0.0001 ^
    --img-size 512 ^
    --gpu 0 ^
    --use-batchnorm ^
    --weighted-loss ^
    --experiment "baseline"

pause
