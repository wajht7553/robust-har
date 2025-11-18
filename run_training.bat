@echo off
REM Quick-start script for LOSO training experiments

echo ================================================================================
echo LOSO Training Framework for Human Activity Recognition
echo ================================================================================
echo.

REM Check if data exists
if not exist "dataset\processed_acc_gyr\X.npy" (
    echo ERROR: Processed data not found!
    echo Please run: python preprocess.py
    echo.
    pause
    exit /b 1
)

echo Data found. Starting training...
echo.

REM Parse command line arguments
set MODEL=%1
set EPOCHS=%2
set BATCH_SIZE=%3

REM Set defaults
if "%MODEL%"=="" set MODEL=both
if "%EPOCHS%"=="" set EPOCHS=50
if "%BATCH_SIZE%"=="" set BATCH_SIZE=32

echo Configuration:
echo   Model: %MODEL%
echo   Epochs: %EPOCHS%
echo   Batch Size: %BATCH_SIZE%
echo.

REM Run training
python train_loso.py --model %MODEL% --epochs %EPOCHS% --batch_size %BATCH_SIZE%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================================================
    echo Training completed successfully!
    echo Check the results/ directory for outputs.
    echo ================================================================================
    echo.
    echo To visualize results, run:
    echo   python visualize_results.py results/[model_name_timestamp]/results.json
    echo.
) else (
    echo.
    echo ================================================================================
    echo Training failed with error code %ERRORLEVEL%
    echo ================================================================================
    echo.
)

pause
