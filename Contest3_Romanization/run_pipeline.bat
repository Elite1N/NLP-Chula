@echo off

echo ==== Traning Baseline T5 (Gold Only) ====
cd src
call conda run --no-capture-output -n nlp_course python train_t5.py --model_name t5_small_char
echo.

echo ==== Training T5 (Gold + Silver) ====
call conda run --no-capture-output -n nlp_course python train_t5.py --model_name t5_silver_augmented --use_silver
echo.

echo ==== Generating Predictions for test set (Using Silver Augmented) ====
call conda run --no-capture-output -n nlp_course python generate.py --model_path ../experiments/t5_silver_augmented/best_model --output_txt ../romanization-test-pred.txt
echo.

echo ==== Checking row length validity ====
cd ..
call conda run --no-capture-output -n nlp_course python scripts/check.py
echo.

echo Pipeline done! You can use the results in experiments/ to draft your report.
