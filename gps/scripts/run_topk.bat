@echo off
setlocal enabledelayedexpansion
echo Running RS Adversarial Detection Experiments with Gradient Strategy and Multiple TOP_N values - bert-attack across Multiple Datasets

REM Create base output directory if it doesn't exist
if not exist "output" mkdir "output"

echo.
echo ************************************************************************
echo Running Gradient experiments with multiple TOP_N values - bert-attack only
echo ************************************************************************

REM Loop through TOP_N values
FOR %%K IN (5 10 15 20 25 30 35 40 45 50) DO (
    set TOP_N=%%K
    echo.
    echo ************************************************************************
    echo Running experiments with TOP_N value: !TOP_N!
    echo ************************************************************************

    REM Loop through datasets
    FOR %%D IN (imdb ag_news yelp) DO (
        set DATASET=%%D
        set MODEL_NAME=roberta_%%D
        REM Base directory, attack name will be appended
        set BASE_DATA_DIR=data/%%D/roberta
        REM Base output directory, subdirs created by script
        set BASE_OUTPUT_DIR=output

        echo.
        echo ******************************************************************
        echo Running experiments with dataset: %%D, TOP_N: !TOP_N!
        echo ******************************************************************
        echo.
        echo *** Model: !MODEL_NAME! ***
        echo *** Base Data Dir: !BASE_DATA_DIR! ***
        echo *** Base Output Dir: !BASE_OUTPUT_DIR! ***
        echo *** Strategy: Gradient  ***
        echo *** TOP_N: !TOP_N! ***

        REM ======================= bert-attack Attack Only =======================
        echo.
        echo #####################################################################
        echo ### Running Attack: bert-attack [Dataset: %%D, TOP_N: !TOP_N!]
        echo #####################################################################

        REM Set the full data directory for bert-attack attack
        set DATA_DIR=!BASE_DATA_DIR!/bert-attack

        REM Check if data directory exists
        if not exist "!DATA_DIR!" (
            echo WARNING: Data directory not found for bert-attack attack at !DATA_DIR!. Skipping...
        ) else (
            echo *** Current Data Dir: !DATA_DIR! ***

            REM Run with Gradient strategy
            echo.
            echo =====================================================================
            echo Running with Gradient strategy [Attack: bert-attack, Dataset: %%D, TOP_N: !TOP_N!]
            echo =====================================================================
            python -m gps --model_name !MODEL_NAME! --data_dir "!DATA_DIR!" --output_dir !BASE_OUTPUT_DIR! --use_saliency --top_n !TOP_N! --distance_metric cosine --detection_methods bilstm --use_importance_channel --filter_importance_scores
            echo Gradient completed for bert-attack attack with dataset %%D, TOP_N: !TOP_N!!
        )
        
        echo.
        echo bert-attack Gradient experiments completed for dataset %%D with TOP_N: !TOP_N!!
    )
)

echo.
echo All bert-attack Gradient experiments completed for all datasets and TOP_N values!
endlocal 