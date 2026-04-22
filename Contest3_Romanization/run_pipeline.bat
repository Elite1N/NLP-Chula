@echo off
echo ==== Traning Baseline T5 (Gold Only) ====
call conda activate nlp_course
cd src
python train_t5.py --model_name t5_small_char
echo.

echo ==== Training T5 (Gold + Silver) ====
python train_t5.py --model_name t5_silver_augmented --use_silver
echo.

echo ==== Generating Predictions for test set (Using Silver Augmented) ====
python generate.py --model_path ../experiments/t5_silver_augmented/best_model --output_txt ../romanization-test-pred.txt
echo.

echo ==== Checking row length validity ====
cd ..
python scripts/check.py
echo.

echo Pipeline done! You can use the results in experiments/ to draft your report.
