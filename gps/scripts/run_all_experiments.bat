@echo off
setlocal enabledelayedexpansion
echo Running All Adversarial Detection Experiments for RoBERTa and DeBERTa Models with fixed TOP_N values

REM Create main output directory if it doesn't exist
if not exist "output" mkdir "output"

REM Create a log file for the run
set LOG_FILE=run_all_log_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%.txt
set LOG_FILE=%LOG_FILE: =0%
echo Starting experiments with RoBERTa and DeBERTa models with fixed TOP_N values (AG News: 25, IMDB: 25, Yelp: 25) at %date% %time% > %LOG_FILE%

echo =====================================================================
echo Starting AG News Experiments (RoBERTa and DeBERTa, TOP_N=25)
echo =====================================================================
echo [%date% %time%] Running AG News experiments with RoBERTa and DeBERTa (TOP_N=25) >> %LOG_FILE%

REM Run AG News experiments using CALL with /B to prevent early exit
CALL run_experiments_ag_news.bat
if errorlevel 1 (
    echo AG News experiments failed with error level !errorlevel! >> %LOG_FILE%
    echo AG News experiments failed with error level !errorlevel!
) else (
    echo.
    echo AG News experiments completed!
    echo [%date% %time%] AG News experiments completed for all model variants >> %LOG_FILE%
    echo.
)

echo =====================================================================
echo Starting IMDB Experiments (RoBERTa and DeBERTa, TOP_N=25)
echo =====================================================================
echo [%date% %time%] Running IMDB experiments with RoBERTa and DeBERTa (TOP_N=25) >> %LOG_FILE%

REM Run IMDB experiments
CALL run_experiments_imdb.bat
if errorlevel 1 (
    echo IMDB experiments failed with error level !errorlevel! >> %LOG_FILE%
    echo IMDB experiments failed with error level !errorlevel!
) else (
    echo.
    echo IMDB experiments completed!
    echo [%date% %time%] IMDB experiments completed for all model variants >> %LOG_FILE%
    echo.
)

echo =====================================================================
echo Starting Yelp Experiments (RoBERTa and DeBERTa, TOP_N=25)
echo =====================================================================
echo [%date% %time%] Running Yelp experiments with RoBERTa and DeBERTa (TOP_N=25) >> %LOG_FILE%

REM Run yelp experiments
CALL run_experiments_yelp.bat
if errorlevel 1 (
    echo Yelp experiments failed with error level !errorlevel! >> %LOG_FILE%
    echo Yelp experiments failed with error level !errorlevel!
) else (
    echo.
    echo Yelp experiments completed!
    echo [%date% %time%] Yelp experiments completed for all model variants >> %LOG_FILE%
    echo.
)

echo All experiments completed for RoBERTa and DeBERTa models!
echo [%date% %time%] All experiments completed for all model variants >> %LOG_FILE%
echo.
echo Check %LOG_FILE% for run log.
echo.
pause 
endlocal 