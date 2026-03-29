@echo off
REM Run k-fold cross validation - train on all 6 folds

REM Activate virtual environment
call ..\.venv\Scripts\activate.bat

set DATA_ROOT=%~dp0..\Full_data_converted
if not "%SEG_DATA_ROOT%"=="" set DATA_ROOT=%SEG_DATA_ROOT%

set METADATA=%DATA_ROOT%\metadata.csv
set FOLDS=%DATA_ROOT%\cv_folds.csv

REM Train on each fold
FOR %%f IN (0 1 2 3 4 5) DO (
    echo.
    echo ========================================
    echo Training fold %%f
    echo ========================================
    echo.
    
    python src\train.py ^
        --metadata "%METADATA%" ^
        --folds-csv "%FOLDS%" ^
        --data-root "%DATA_ROOT%" ^
        --fold %%f ^
        --epochs 15 ^
        --batch-size 4 ^
        --lr 0.0001 ^
        --img-size 512 ^
        --gpu 0 ^
        --use-batchnorm ^
        --weighted-loss ^
        --experiment "baseline"
    
    IF ERRORLEVEL 1 (
        echo Error in fold %%f
        pause
        exit /b 1
    )
)

echo.
echo ========================================
echo All folds completed!
echo ========================================
pause
