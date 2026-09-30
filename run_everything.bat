@echo off
title Department Store Reward ML - One Click Run
color 0B
cls
echo.
echo  ============================================================
echo   AI-Powered Customer Intelligence
echo   Personalised Reward Recommendation
echo   ONE-CLICK RUN
echo  ============================================================
echo.

REM ---- find Python -------------------------------------------------------
set PY=
where python >nul 2>nul && set PY=python
if "%PY%"=="" where py >nul 2>nul && set PY=py -3
if "%PY%"=="" (
    echo  [ERROR] Python was not found on this computer.
    echo.
    echo  Install Python 3 from https://www.python.org/downloads/
    echo  IMPORTANT: tick "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)
echo  [1/6] Python found: & %PY% --version
echo.

cd /d "%~dp0"

REM ---- install missing packages -------------------------------------------
echo  [2/6] Checking packages (this may take a minute on first run)...
echo.
%PY% -m pip install -q pandas numpy scikit-learn matplotlib xgboost catboost nbformat
if errorlevel 1 (
    echo  [WARNING] Some packages may be missing. Trying again with full output...
    %PY% -m pip install pandas numpy scikit-learn matplotlib xgboost catboost nbformat
)
echo      Done.
echo.

REM ---- generate data ------------------------------------------------------
echo  [3/6] Generating the dataset...
echo.
%PY% src\data_generator.py
if errorlevel 1 (
    echo  [ERROR] Data generation failed. See messages above.
    pause
    exit /b 1
)
echo.

REM ---- run the ML pipeline ------------------------------------------------
echo  [4/6] Training models and running the reward engine
echo        (about 30-60 seconds, please wait)...
echo.
%PY% run_pipeline.py
if errorlevel 1 (
    echo  [ERROR] Pipeline failed. See messages above.
    pause
    exit /b 1
)
echo.

REM ---- build the presentation report --------------------------------------
echo  [5/6] Building the presentation report...
echo.
%PY% src\report_builder.py
if errorlevel 1 (
    echo  [ERROR] Report build failed. See messages above.
    pause
    exit /b 1
)
echo.

REM ---- open the result ----------------------------------------------------
echo  [6/6] Opening the report in your browser...
echo.
if exist "outputs\report\project_report.html" (
    start "" "outputs\report\project_report.html"
) else (
    echo  [WARNING] Report file was not found at outputs\report\project_report.html
)
echo.
echo  ============================================================
echo   ALL DONE
echo.
echo   Report : outputs\report\project_report.html
echo   Charts : outputs\figures\
echo   Tables : outputs\reports\
echo   Model  : models\
echo.
echo   The report opens in your browser. You can print it,
echo   save it as PDF, or email it to whoever needs it.
echo  ============================================================
echo.
pause
