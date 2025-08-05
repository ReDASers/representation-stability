@echo off
SETLOCAL EnableDelayedExpansion
REM ======================================================================
REM SHARPNESS-BASED GENERALIZATION EXPERIMENTS (R1-R3)
REM using extended sharpness_detector.py
REM ======================================================================
REM This script runs generalization experiments for sharpness detection
REM matching the experiments from other detection methods
REM ======================================================================

REM Set common variables
SET TARGET_ATTACKS=textfooler,deepwordbug,bert-attack
SET TF_ATTACK=textfooler
SET DWB_ATTACK=deepwordbug
SET BA_ATTACK=bert-attack

SET YELP_DATASET=yelp
SET IMDB_DATASET=imdb
SET YELP_IMDB_DATASETS=yelp,imdb

REM Base output directory
SET BASE_OUTPUT_DIR=%CD%\output\sharpness_generalization_results
IF NOT EXIST "%BASE_OUTPUT_DIR%" mkdir "%BASE_OUTPUT_DIR%"

ECHO ======================================================================
ECHO Starting Sharpness-Based Generalization Studies (R1-R3)
ECHO Using extended sharpness_detector.py
ECHO Using fixed seed 42.
ECHO ======================================================================
ECHO.

REM --- Common parameters for sharpness_detector.py ---
SET COMMON_PARAMS=--data_dir data

REM ======================================================================
REM R1(a): Dataset shift (Yelp -> IMDB)
REM ======================================================================
ECHO Running R1a: Dataset shift (Yelp to IMDB)
SET R1A_OUT_DIR=%BASE_OUTPUT_DIR%\R1a_Yelp_to_IMDB
IF NOT EXIST "%R1A_OUT_DIR%" mkdir "%R1A_OUT_DIR%"
python sharpness\sharpness_detector.py %COMMON_PARAMS% ^
    --experiment_type cross_dataset ^
    --train_datasets %YELP_DATASET% ^
    --train_models roberta ^
    --train_attacks %TARGET_ATTACKS% ^
    --test_datasets %IMDB_DATASET% ^
    --test_models roberta ^
    --test_attacks %TARGET_ATTACKS% ^
    --output_dir "%R1A_OUT_DIR%"
ECHO R1a Complete. Results in %R1A_OUT_DIR%
ECHO.

REM ======================================================================
REM R1(b): Dataset shift (IMDB -> Yelp)
REM ======================================================================
ECHO Running R1b: Dataset shift (IMDB to Yelp)
SET R1B_OUT_DIR=%BASE_OUTPUT_DIR%\R1b_IMDB_to_Yelp
IF NOT EXIST "%R1B_OUT_DIR%" mkdir "%R1B_OUT_DIR%"
python sharpness\sharpness_detector.py %COMMON_PARAMS% ^
    --experiment_type cross_dataset ^
    --train_datasets %IMDB_DATASET% ^
    --train_models roberta ^
    --train_attacks %TARGET_ATTACKS% ^
    --test_datasets %YELP_DATASET% ^
    --test_models roberta ^
    --test_attacks %TARGET_ATTACKS% ^
    --output_dir "%R1B_OUT_DIR%"
ECHO R1b Complete. Results in %R1B_OUT_DIR%
ECHO.

REM ======================================================================
REM R2(a): Attack shift (TextFooler -> DeepWordBug)
REM ======================================================================
ECHO Running R2a: Attack shift (TextFooler to DeepWordBug)
SET R2A_OUT_DIR=%BASE_OUTPUT_DIR%\R2a_TF_to_DWB
IF NOT EXIST "%R2A_OUT_DIR%" mkdir "%R2A_OUT_DIR%"
python sharpness\sharpness_detector.py %COMMON_PARAMS% ^
    --experiment_type cross_attack ^
    --train_datasets %YELP_IMDB_DATASETS% ^
    --train_models roberta ^
    --train_attacks %TF_ATTACK% ^
    --test_attacks %DWB_ATTACK% ^
    --output_dir "%R2A_OUT_DIR%"
ECHO R2a Complete. Results in %R2A_OUT_DIR%
ECHO.

REM ======================================================================
REM R2(b): Attack shift (DeepWordBug -> TextFooler)
REM ======================================================================
ECHO Running R2b: Attack shift (DeepWordBug to TextFooler)
SET R2B_OUT_DIR=%BASE_OUTPUT_DIR%\R2b_DWB_to_TF
IF NOT EXIST "%R2B_OUT_DIR%" mkdir "%R2B_OUT_DIR%"
python sharpness\sharpness_detector.py %COMMON_PARAMS% ^
    --experiment_type cross_attack ^
    --train_datasets %YELP_IMDB_DATASETS% ^
    --train_models roberta ^
    --train_attacks %DWB_ATTACK% ^
    --test_attacks %TF_ATTACK% ^
    --output_dir "%R2B_OUT_DIR%"
ECHO R2b Complete. Results in %R2B_OUT_DIR%
ECHO.

REM ======================================================================
REM R2(c): Attack shift (TextFooler -> BERT-Attack)
REM ======================================================================
ECHO Running R2c: Attack shift (TextFooler to BERT-Attack)
SET R2C_OUT_DIR=%BASE_OUTPUT_DIR%\R2c_TF_to_BA
IF NOT EXIST "%R2C_OUT_DIR%" mkdir "%R2C_OUT_DIR%"
python sharpness\sharpness_detector.py %COMMON_PARAMS% ^
    --experiment_type cross_attack ^
    --train_datasets %YELP_IMDB_DATASETS% ^
    --train_models roberta ^
    --train_attacks %TF_ATTACK% ^
    --test_attacks %BA_ATTACK% ^
    --output_dir "%R2C_OUT_DIR%"
ECHO R2c Complete. Results in %R2C_OUT_DIR%
ECHO.

REM ======================================================================
REM R2(d): Attack shift (BERT-Attack -> TextFooler)
REM ======================================================================
ECHO Running R2d: Attack shift (BERT-Attack to TextFooler)
SET R2D_OUT_DIR=%BASE_OUTPUT_DIR%\R2d_BA_to_TF
IF NOT EXIST "%R2D_OUT_DIR%" mkdir "%R2D_OUT_DIR%"
python sharpness\sharpness_detector.py %COMMON_PARAMS% ^
    --experiment_type cross_attack ^
    --train_datasets %YELP_IMDB_DATASETS% ^
    --train_models roberta ^
    --train_attacks %BA_ATTACK% ^
    --test_attacks %TF_ATTACK% ^
    --output_dir "%R2D_OUT_DIR%"
ECHO R2d Complete. Results in %R2D_OUT_DIR%
ECHO.

REM ======================================================================
REM R2(e): Attack shift (DeepWordBug -> BERT-Attack)
REM ======================================================================
ECHO Running R2e: Attack shift (DeepWordBug to BERT-Attack)
SET R2E_OUT_DIR=%BASE_OUTPUT_DIR%\R2e_DWB_to_BA
IF NOT EXIST "%R2E_OUT_DIR%" mkdir "%R2E_OUT_DIR%"
python sharpness\sharpness_detector.py %COMMON_PARAMS% ^
    --experiment_type cross_attack ^
    --train_datasets %YELP_IMDB_DATASETS% ^
    --train_models roberta ^
    --train_attacks %DWB_ATTACK% ^
    --test_attacks %BA_ATTACK% ^
    --output_dir "%R2E_OUT_DIR%"
ECHO R2e Complete. Results in %R2E_OUT_DIR%
ECHO.

REM ======================================================================
REM R2(f): Attack shift (BERT-Attack -> DeepWordBug)
REM ======================================================================
ECHO Running R2f: Attack shift (BERT-Attack to DeepWordBug)
SET R2F_OUT_DIR=%BASE_OUTPUT_DIR%\R2f_BA_to_DWB
IF NOT EXIST "%R2F_OUT_DIR%" mkdir "%R2F_OUT_DIR%"
python sharpness\sharpness_detector.py %COMMON_PARAMS% ^
    --experiment_type cross_attack ^
    --train_datasets %YELP_IMDB_DATASETS% ^
    --train_models roberta ^
    --train_attacks %BA_ATTACK% ^
    --test_attacks %DWB_ATTACK% ^
    --output_dir "%R2F_OUT_DIR%"
ECHO R2f Complete. Results in %R2F_OUT_DIR%
ECHO.

REM ======================================================================
REM R3(a): Encoder shift (RoBERTa -> DeBERTa)
REM ======================================================================
ECHO Running R3a: Encoder shift (RoBERTa to DeBERTa)
SET R3A_OUT_DIR=%BASE_OUTPUT_DIR%\R3a_RoBERTa_to_DeBERTa
IF NOT EXIST "%R3A_OUT_DIR%" mkdir "%R3A_OUT_DIR%"
python sharpness\sharpness_detector.py %COMMON_PARAMS% ^
    --experiment_type cross_encoder ^
    --train_datasets %YELP_IMDB_DATASETS% ^
    --train_models roberta ^
    --train_attacks %TARGET_ATTACKS% ^
    --test_models deberta ^
    --test_attacks %TARGET_ATTACKS% ^
    --output_dir "%R3A_OUT_DIR%"
ECHO R3a Complete. Results in %R3A_OUT_DIR%
ECHO.

REM ======================================================================
REM R3(b): Encoder shift (DeBERTa -> RoBERTa)
REM ======================================================================
ECHO Running R3b: Encoder shift (DeBERTa to RoBERTa)
SET R3B_OUT_DIR=%BASE_OUTPUT_DIR%\R3b_DeBERTa_to_RoBERTa
IF NOT EXIST "%R3B_OUT_DIR%" mkdir "%R3B_OUT_DIR%"
python sharpness\sharpness_detector.py %COMMON_PARAMS% ^
    --experiment_type cross_encoder ^
    --train_datasets %YELP_IMDB_DATASETS% ^
    --train_models deberta ^
    --train_attacks %TARGET_ATTACKS% ^
    --test_models roberta ^
    --test_attacks %TARGET_ATTACKS% ^
    --output_dir "%R3B_OUT_DIR%"
ECHO R3b Complete. Results in %R3B_OUT_DIR%
ECHO.

ECHO ======================================================================
ECHO All sharpness-based generalization experiments completed!
ECHO Results are saved in: %BASE_OUTPUT_DIR%
ECHO ======================================================================
:EOF
ENDLOCAL 