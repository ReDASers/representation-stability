@echo off
setlocal enabledelayedexpansion
echo Starting sharpness detection experiments for all combinations...

set TOTAL_EXPERIMENTS=0
set SUCCESSFUL_EXPERIMENTS=0
set FAILED_EXPERIMENTS=0

set DATASETS=imdb yelp ag_news
set MODELS=roberta deberta
set ATTACKS=textfooler bert-attack deepwordbug

set OUTPUT_BASE_DIR=output\sharpness_results

for %%d in (%DATASETS%) do (
    for %%m in (%MODELS%) do (
        for %%a in (%ATTACKS%) do (
            echo.
            echo ======================================================================
            echo Running sharpness detector for %%d / %%m / %%a
            echo ======================================================================
            
            set /a TOTAL_EXPERIMENTS+=1
            
            python sharpness\sharpness_detector.py ^
                --data_dir data ^
                --dataset %%d ^
                --model %%m ^
                --attack %%a ^
                --output_dir "%OUTPUT_BASE_DIR%"
            
            if !ERRORLEVEL! neq 0 (
                echo ERROR: Sharpness detection failed for %%d / %%m / %%a with exit code !ERRORLEVEL!
                echo Continuing with next experiment...
                set /a FAILED_EXPERIMENTS+=1
            ) else (
                echo Successfully completed sharpness detection for %%d / %%m / %%a
                echo Results saved to "%OUTPUT_BASE_DIR%\%%d\%%m\%%a\sharpness\results\"
                set /a SUCCESSFUL_EXPERIMENTS+=1
            )
        )
    )
)

echo.
echo ======================================================================
echo All sharpness detection experiments complete.
echo Total experiments: !TOTAL_EXPERIMENTS!
echo Successful: !SUCCESSFUL_EXPERIMENTS!
echo Failed: !FAILED_EXPERIMENTS!
echo ======================================================================
endlocal 