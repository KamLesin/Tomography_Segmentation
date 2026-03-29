@echo off
REM Run k-fold cross validation on multiple GPUs in parallel
REM Each fold runs on a separate GPU simultaneously

REM Activate virtual environment
call ..\.venv\Scripts\activate.bat

set DATA_ROOT=%~dp0..\Full_data_converted
if not "%SEG_DATA_ROOT%"=="" set DATA_ROOT=%SEG_DATA_ROOT%

set METADATA=%DATA_ROOT%\metadata.csv
set FOLDS=%DATA_ROOT%\cv_folds.csv

echo.
echo ========================================
echo Starting 6-fold CV in PARALLEL mode
echo Each fold will run on a separate GPU
echo ========================================
echo.

REM Start each fold on a different GPU in background
FOR %%f IN (0 1 2 3 4 5) DO (
    echo Starting fold %%f on GPU %%f...
    
    start "Training Fold %%f - GPU %%f" python src\train.py ^
        --metadata "%METADATA%" ^
        --folds-csv "%FOLDS%" ^
        --data-root "%DATA_ROOT%" ^
        --fold %%f ^
        --epochs 15 ^
        --batch-size 4 ^
        --lr 0.0001 ^
        --img-size 512 ^
        --gpu %%f ^
        --use-batchnorm ^
        --weighted-loss ^
        --experiment "baseline"
)

echo.
echo ========================================
echo All 6 folds started in PARALLEL!
echo ========================================
echo.
echo Monitor progress:
echo - Each fold runs in a separate window
echo - Training Fold 0 - GPU 0
echo - Training Fold 1 - GPU 1
echo - Training Fold 2 - GPU 2
echo - Training Fold 3 - GPU 3
echo - Training Fold 4 - GPU 4
echo - Training Fold 5 - GPU 5
echo.
echo Expected completion time: ~1/6 of sequential version
echo (if you have 6 GPUs available)
echo.
echo After all folds complete, run:
echo   python aggregate_results.py --experiment-dir runs/baseline
echo.
pause
