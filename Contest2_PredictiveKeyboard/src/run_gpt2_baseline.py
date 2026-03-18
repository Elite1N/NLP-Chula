import os
import pandas as pd
import argparse
from gpt2_model import GPT2Predictor
from utils import evaluate_model, generate_test_predictions, log_experiment_result

# Conf (Adjusted for running from project root or src/)
if os.path.exists('data/dev_set.csv'):
    DEV_PATH = 'data/dev_set.csv'
    TEST_PATH_NO_ANSWER = 'data/test_set_no_answer.csv'
    EXPERIMENT_DIR = 'experiments/gpt2_baseline'
else:
    # Fallback to src-relative paths
    DEV_PATH = '../data/dev_set.csv'
    TEST_PATH_NO_ANSWER = '../data/test_set_no_answer.csv'
    EXPERIMENT_DIR = '../experiments/gpt2_baseline'
    
OUTPUT_PRED_PATH = os.path.join(EXPERIMENT_DIR, 'test_set_pred.txt')

def main():
    parser = argparse.ArgumentParser(description="Run GPT-2 Baseline")
    parser.add_argument('--limit_dev', type=int, default=None, help="Limit number of dev examples for quick testing")
    args = parser.parse_args()

    # 1. Load Data (No training data needed for pre-trained GPT-2)
    print("Loading datasets...")
    dev_df = pd.read_csv(DEV_PATH)
    test_df = pd.read_csv(TEST_PATH_NO_ANSWER)

    if args.limit_dev:
        print(f"Limiting dev set to {args.limit_dev} examples for testing...")
        dev_df = dev_df.head(args.limit_dev)

    # 2. Init Model
    model = GPT2Predictor(model_name='gpt2') # uses 'gpt2' (small) by default

    # 3. Evaluate
    print(f"Evaluating GPT-2 on {len(dev_df)} examples...")
    accuracy = evaluate_model(model, dev_df)
    
    # Save Metrics
    log_experiment_result(
        EXPERIMENT_DIR, 
        'GPT-2 (Pre-trained)', 
        'Model=gpt2-small, TopK=1000', 
        'Pre-trained (WebText)', 
        accuracy
    )

    # 4. Generate Predictions for submission (Full test set)
    # WARNING: This might take a while on CPU.
    # generate_test_predictions(model, test_df, OUTPUT_PRED_PATH)
    print("\nSkipping test prediction generation by default to save time. Uncomment in script if needed.")
    
    # If users wants to generate predictions, they can uncomment the line above.
    # Or we can make it an argument. For now, let's keep it safe.

if __name__ == "__main__":
    main()
