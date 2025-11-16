@echo off
REM Batch script to train DANN models for all cancer domains sequentially
REM This script trains models for Breast, Colon, Kidney, and Lung cancer datasets

echo ================================================================================
echo TRAINING DOMAIN ADAPTATION MODELS FOR ALL CANCER DOMAINS
echo ================================================================================
echo.

REM Set start time
set START_TIME=%TIME%
echo Start Time: %START_TIME%
echo.

REM Train Breast Cancer model
echo --------------------------------------------------------------------------------
echo [1/4] Training Breast Cancer Model
echo --------------------------------------------------------------------------------
python scripts/run_training.py experiment_configs/dann_train_val_breast_cancer_config.yaml
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Breast Cancer training failed with error code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)
echo.
echo Breast Cancer training completed successfully!
echo.

REM Train Colon Cancer model
echo --------------------------------------------------------------------------------
echo [2/4] Training Colon Cancer Model
echo --------------------------------------------------------------------------------
python scripts/run_training.py experiment_configs/dann_train_val_colon_cancer_config.yaml
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Colon Cancer training failed with error code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)
echo.
echo Colon Cancer training completed successfully!
echo.

REM Train Kidney Cancer model
echo --------------------------------------------------------------------------------
echo [3/4] Training Kidney Cancer Model
echo --------------------------------------------------------------------------------
python scripts/run_training.py experiment_configs/dann_train_val_kidney_cancer_config.yaml
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Kidney Cancer training failed with error code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)
echo.
echo Kidney Cancer training completed successfully!
echo.

REM Train Lung Cancer model
echo --------------------------------------------------------------------------------
echo [4/4] Training Lung Cancer Model
echo --------------------------------------------------------------------------------
python scripts/run_training.py experiment_configs/dann_train_val_lung_cancer_config.yaml
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Lung Cancer training failed with error code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)
echo.
echo Lung Cancer training completed successfully!
echo.

REM Set end time
set END_TIME=%TIME%

echo ================================================================================
echo ALL TRAINING COMPLETED SUCCESSFULLY!
echo ================================================================================
echo Start Time: %START_TIME%
echo End Time:   %END_TIME%
echo.
echo Trained models saved in:
echo   - results/dann_train_val_breast_cancer_results/
echo   - results/dann_train_val_colon_cancer_results/
echo   - results/dann_train_val_kidney_cancer_results/
echo   - results/dann_train_val_lung_cancer_results/
echo.

pause
