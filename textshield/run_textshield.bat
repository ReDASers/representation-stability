@echo off
setlocal enabledelayedexpansion
echo Starting detector training and evaluation for all combinations...

set DATASETS=imdb yelp ag_news
set MODELS=roberta deberta
set ATTACKS=textfooler bert-attack deepwordbug
set STRATEGIES=awi

set OUTPUT_BASE_DIR=output\detector_results

for %%d in (%DATASETS%) do (
    for %%m in (%MODELS%) do (
        for %%a in (%ATTACKS%) do (
            for %%s in (%STRATEGIES%) do (
                echo.
                echo ======================================================================
                echo Training detector for %%d / %%m / %%a / %%s
                echo ======================================================================
                
                set OUTPUT_DIR=%OUTPUT_BASE_DIR%\%%d\%%m\%%a\%%s
                
                python textshield/train_evaluate_detector.py ^
                    --features_dir data ^
                    --dataset %%d ^
                    --model %%m ^
                    --attack %%a ^
                    --strategy %%s ^
                    --output_dir "!OUTPUT_DIR!"
                
                if !ERRORLEVEL! neq 0 (
                    echo WARNING: Detector training failed for %%d / %%m / %%a / %%s
                ) else (
                    echo Successfully completed detector for %%d / %%m / %%a / %%s
                )
            )
        )
    )
)

echo.
echo All detector training and evaluation complete.
endlocal 