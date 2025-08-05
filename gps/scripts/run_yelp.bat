@echo off
setlocal enabledelayedexpansion
echo Running Adversarial Detection Experiments with Multiple Word Selection Strategies - Yelp Dataset

REM Perturbation strategy no longer configurable - removed from refactoring

REM Create base output directory if it doesn't exist
if not exist "output" mkdir "output"

REM Set fixed TOP_N value for Yelp dataset
set TOP_N=25

echo.
echo ************************************************************************
echo Running experiments with TOP_N value: %TOP_N%
echo ************************************************************************

REM Loop through model variants
FOR %%M IN (roberta deberta) DO (
    set MODEL_NAME=%%M_yelp
    REM Base directory, attack name will be appended
    set BASE_DATA_DIR=data/yelp/%%M
    REM Base output directory, subdirs created by script
    set BASE_OUTPUT_DIR=output

    echo.
    echo ******************************************************************
    echo Running experiments with model variant: %%M, TOP_N: %TOP_N%
    echo ******************************************************************
    echo.
    echo *** Model: !MODEL_NAME! ***
    echo *** Base Data Dir: !BASE_DATA_DIR! ***
    echo *** Base Output Dir: !BASE_OUTPUT_DIR! ***
    echo *** TOP_N: %TOP_N% ***

    REM ======================= Attack Loop =======================
    FOR %%A IN (textfooler deepwordbug bert-attack) DO (
        echo.
        echo #####################################################################
        echo ### Running Attack: %%A [Model: %%M, TOP_N: %TOP_N%]
        echo #####################################################################

        REM Set the full data directory for the current attack
        set DATA_DIR=!BASE_DATA_DIR!/%%A

        REM Check if data directory exists
        if not exist "!DATA_DIR!" (
            echo WARNING: Data directory not found for attack %%A at !DATA_DIR!. Skipping...
        ) else (
            echo *** Current Data Dir: !DATA_DIR! ***

            REM Run with Saliency strategy
            echo.
            echo =====================================================================
            echo Running with Saliency strategy [Attack: %%A, Model: %%M, TOP_N: %TOP_N%]
            echo =====================================================================
            python -m gps --model_name !MODEL_NAME! --data_dir "!DATA_DIR!" --output_dir !BASE_OUTPUT_DIR! --use_saliency --top_n %TOP_N% --distance_metric cosine --detection_methods bilstm --use_importance_channel --filter_importance_scores
            echo Saliency completed for attack %%A with model %%M, TOP_N: %TOP_N%!

            REM Run with Gradient × Attention strategy
            echo =====================================================================
            echo Running with Gradient x Attention strategy [Attack: %%A, Model: %%M, TOP_N: %TOP_N%]
            echo =====================================================================
            python -m gps --model_name !MODEL_NAME! --data_dir "!DATA_DIR!" --output_dir !BASE_OUTPUT_DIR! --use_gradient_attention --top_n %TOP_N% --distance_metric cosine --detection_methods bilstm --use_importance_channel --filter_importance_scores
            echo Gradient x Attention completed for attack %%A with model %%M, TOP_N: %TOP_N%!

            REM Run with Attention strategy
            echo =====================================================================
            echo Running with Attention strategy [Attack: %%A, Model: %%M, TOP_N: %TOP_N%]
            echo =====================================================================
            python -m gps --model_name !MODEL_NAME! --data_dir "!DATA_DIR!" --output_dir !BASE_OUTPUT_DIR! --use_attention --top_n %TOP_N% --distance_metric cosine --detection_methods bilstm --use_importance_channel --filter_importance_scores
            echo Attention completed for attack %%A with model %%M, TOP_N: %TOP_N%!

            REM Run with Random strategy
            echo.
            echo =====================================================================
            echo Running with Random strategy [Attack: %%A, Model: %%M, TOP_N: %TOP_N%]
            echo =====================================================================
            python -m gps --model_name !MODEL_NAME! --data_dir "!DATA_DIR!" --output_dir !BASE_OUTPUT_DIR! --use_random --top_n %TOP_N% --random_seed 42 --distance_metric cosine --detection_methods bilstm --use_importance_channel --filter_importance_scores
            echo Random completed for attack %%A with model %%M, TOP_N: %TOP_N%!
        )
    )
    
    echo.
    echo All Yelp experiments completed for model variant %%M with TOP_N: %TOP_N%!
)

echo.
echo All Yelp experiments completed for all model variants!
endlocal 