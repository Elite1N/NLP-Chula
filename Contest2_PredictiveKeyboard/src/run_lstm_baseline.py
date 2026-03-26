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
    args = parser.parse_args()

    # Conf
    if os.path.exists('experiments/lstm_baseline'):
        EXPERIMENT_DIR = 'experiments/lstm_baseline'
        DEV_PATH = 'data/dev_set.csv'
        TEST_PATH_NO_ANSWER = 'data/test_set_no_answer.csv'
    else:
        EXPERIMENT_DIR = '../experiments/lstm_baseline'
        DEV_PATH = '../data/dev_set.csv'
        TEST_PATH_NO_ANSWER = '../data/test_set_no_answer.csv'
        
    MODEL_PATH = os.path.join(EXPERIMENT_DIR, 'lstm_model.pth')
    VOCAB_PATH = os.path.join(EXPERIMENT_DIR, 'vocab.pkl')
    OUTPUT_PRED_PATH = os.path.join(EXPERIMENT_DIR, 'test_set_pred.txt')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Predictor
    print("Loading LSTM Predictor...")
    # NOTE: Ensure dims match training arguments!
    # For now hardcoded to match defaults in train_lstm.py.
    # In a real pipeline, save args config alongside model.
    predictor = LSTMPredictor(MODEL_PATH, VOCAB_PATH, embed_dim=128, hidden_dim=256, seq_len=5, device=str(device))
    
    # 2. Evaluate
    dev_df = pd.read_csv(DEV_PATH)
    if args.limit_dev:
        dev_df = dev_df.head(args.limit_dev)
        
    print(f"Evaluating LSTM on {len(dev_df)} examples...")
    accuracy = evaluate_model(predictor, dev_df)
    
    # Save Metrics
    log_experiment_result(
        EXPERIMENT_DIR, 
        'LSTM (From Scratch)', 
        'Embed=128, Hidden=256, Seq=5, Epochs=3', 
        '50k lines (Subset)', 
        accuracy
    )
    
    # 3. Predict Test
    print("Generating test predictions...")
    test_df = pd.read_csv(TEST_PATH_NO_ANSWER)
    generate_test_predictions(predictor, test_df, OUTPUT_PRED_PATH)

if __name__ == "__main__":
    main()
