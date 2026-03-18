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
    
    # Custom evaluation loop for batching
    correct = 0
    total = len(dev_df)
    batch_size = 32 # Adjust based on GPU VRAM
    
    contexts = dev_df['context'].tolist()
    first_letters = dev_df['first letter'].tolist()
    answers = dev_df['answer'].tolist()
    
    import time
    start_eval = time.time()
    
    # Predict in batches
    # model.predict_batch handles the loop internally? No, I implemented it to handle a list.
    # So we can just pass the whole list if it handles chunking, or loop here.
    # My implementation of predict_batch handles chunking.
    print(f"Running inference with batch size {batch_size}...")
    predictions = model.predict_batch(contexts, first_letters, batch_size=batch_size)
    
    # Calculate accuracy
    for pred, ans in zip(predictions, answers):
        if pred == ans:
            correct += 1
            
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Evaluation time: {time.time() - start_eval:.2f}s")
    
    # Save Metrics
    log_experiment_result(
        EXPERIMENT_DIR, 
        'GPT-2 (Pre-trained)', 
        f'Model=gpt2-small, TopK=1000, Batch={batch_size}', 
        'Pre-trained (WebText)', 
        accuracy
    )

    # 4. Generate Predictions for submission (Full test set)
    # WARNING: This might take a while on CPU.
    # To run this, uncomment below:
    # print(f"Generating predictions for {len(test_df)} test examples...")
    # test_contexts = test_df['context'].tolist()
    # test_first_letters = test_df['first letter'].tolist()
    # test_preds = model.predict_batch(test_contexts, test_first_letters, batch_size=batch_size)
    # with open(OUTPUT_PRED_PATH, 'w', encoding='utf-8') as f:
    #    for pred in test_preds:
    #        f.write(f"{pred}\n")
    # print(f"Saved predictions to {OUTPUT_PRED_PATH}")
    
    print("\nSkipping test prediction generation by default to save time. Uncomment in script if needed.")
    
    # If users wants to generate predictions, they can uncomment the line above.
    # Or we can make it an argument. For now, let's keep it safe.

if __name__ == "__main__":
    main()
