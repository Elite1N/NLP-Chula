import torch
import os
import argparse
import pandas as pd
from vocab import Vocabulary
from lstm_model import LSTMModel, LSTMPredictor
from utils import evaluate_model, generate_test_predictions, log_experiment_result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit_dev', type=int, default=None)
    parser.add_argument('--no_submission', action='store_true', help='Skip submission generation')
    args = parser.parse_args()
    
    print(f"Args: {args}")

    # Conf
    if os.path.exists('experiments/lstm_full'):
        EXPERIMENT_DIR = 'experiments/lstm_full'
        DEV_PATH = 'data/dev_set.csv'
        TEST_PATH_NO_ANSWER = 'data/test_set_no_answer.csv'
    else:
        EXPERIMENT_DIR = '../experiments/lstm_full'
        DEV_PATH = '../data/dev_set.csv'
        TEST_PATH_NO_ANSWER = '../data/test_set_no_answer.csv'
        
    MODEL_PATH = os.path.join(EXPERIMENT_DIR, 'lstm_model_full.pth')
    VOCAB_PATH = os.path.join(EXPERIMENT_DIR, 'vocab.pkl')
    OUTPUT_PRED_PATH = os.path.join(EXPERIMENT_DIR, 'test_set_pred_full.txt')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Please train it first.")
        return

    # 1. Load Predictor
    print("Loading LSTM Predictor (Full)...")
    # Matches training params in train_lstm_full.py
    predictor = LSTMPredictor(MODEL_PATH, VOCAB_PATH, embed_dim=128, hidden_dim=256, seq_len=5, device=str(device))
    
    # 2. Evaluate
    print(f"Loading Dev Set from {DEV_PATH}...")
    dev_df = pd.read_csv(DEV_PATH)
    if args.limit_dev:
        dev_df = dev_df.head(args.limit_dev)
        
    print(f"Evaluating LSTM on {len(dev_df)} examples...")
    accuracy = evaluate_model(predictor, dev_df)
    print(f"Accuracy: {accuracy}")
    
    # Save Metrics
    log_experiment_result(
        EXPERIMENT_DIR, 
        'LSTM (Full)', 
        'Embed=128, Hidden=256, Seq=5, Epochs=1', 
        'Full Dataset', 
        accuracy,
        filename='evaluations.csv'
    )
    print(f"Logged result to {os.path.join(EXPERIMENT_DIR, 'evaluations.csv')}")
    
    # 3. Generate Submission
    if not args.no_submission:
        print(f"Generating predictions for {TEST_PATH_NO_ANSWER}...")
        test_df = pd.read_csv(TEST_PATH_NO_ANSWER)
        generate_test_predictions(predictor, test_df, OUTPUT_PRED_PATH)
        print(f"Saved predictions to {OUTPUT_PRED_PATH}")

if __name__ == "__main__":
    main()
