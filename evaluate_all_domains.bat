@echo off
REM Batch script to evaluate all trained DANN models on test sets
REM This script evaluates models for Breast, Colon, Kidney, and Lung cancer datasets

echo ================================================================================
echo EVALUATING DOMAIN ADAPTATION MODELS FOR ALL CANCER DOMAINS
echo ================================================================================
echo.

REM Set start time
set START_TIME=%TIME%
echo Start Time: %START_TIME%
echo.

REM Evaluate Breast Cancer model
echo --------------------------------------------------------------------------------
echo [1/4] Evaluating Breast Cancer Model
echo --------------------------------------------------------------------------------
python scripts/run_evaluation.py experiment_configs/dann_test_breast_cancer_config.yaml
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Breast Cancer evaluation failed with error code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)
echo.
echo Breast Cancer evaluation completed successfully!
echo.

REM Evaluate Colon Cancer model
echo --------------------------------------------------------------------------------
echo [2/4] Evaluating Colon Cancer Model
echo --------------------------------------------------------------------------------
python scripts/run_evaluation.py experiment_configs/dann_test_colon_cancer_config.yaml
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Colon Cancer evaluation failed with error code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)
echo.
echo Colon Cancer evaluation completed successfully!
echo.

REM Evaluate Kidney Cancer model
echo --------------------------------------------------------------------------------
echo [3/4] Evaluating Kidney Cancer Model
echo --------------------------------------------------------------------------------
python scripts/run_evaluation.py experiment_configs/dann_test_kidney_cancer_config.yaml
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Kidney Cancer evaluation failed with error code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)
echo.
echo Kidney Cancer evaluation completed successfully!
echo.

REM Evaluate Lung Cancer model
echo --------------------------------------------------------------------------------
echo [4/4] Evaluating Lung Cancer Model
echo --------------------------------------------------------------------------------
python scripts/run_evaluation.py experiment_configs/dann_test_lung_cancer_config.yaml
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Lung Cancer evaluation failed with error code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)
echo.
echo Lung Cancer evaluation completed successfully!
echo.

REM Set end time
set END_TIME=%TIME%

echo ================================================================================
echo ALL EVALUATIONS COMPLETED SUCCESSFULLY!
echo ================================================================================
echo Start Time: %START_TIME%
echo End Time:   %END_TIME%
echo.
echo Evaluation results saved in:
echo   - results/dann_test_breast_cancer_results/
echo   - results/dann_test_colon_cancer_results/
echo   - results/dann_test_kidney_cancer_results/
echo   - results/dann_test_lung_cancer_results/
echo.

pause
