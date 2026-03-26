import os
import argparse
import pandas as pd
from BackoffNGram_baseline import BackoffNGramModel # Renamed from previous tool error
from gpt2_model import GPT2Predictor
from utils import load_training_data, evaluate_model, generate_test_predictions, log_experiment_result

# Conf
TRAIN_PATH = '../data/train.src.tok'
DEV_PATH = '../data/dev_set.csv'
TEST_PATH_NO_ANSWER = '../data/test_set_no_answer.csv'
EXPERIMENT_DIR = '../experiments/hybrid'
MODEL_SAVE_PATH = os.path.join(EXPERIMENT_DIR, 'ngram_full.pkl')
OUTPUT_PRED_PATH = os.path.join(EXPERIMENT_DIR, 'test_set_pred.txt')

class HybridModel:
    def __init__(self, ngram_model, gpt2_model, ngram_trust_threshold=4):
        """
        ngram_trust_threshold: If we find a match in an N-gram model with N >= threshold, we trust it.
                               Otherwise (if we backoff to N < threshold), we consult GPT-2.
        """
        self.ngram_model = ngram_model
        self.gpt2_model = gpt2_model
        self.threshold = ngram_trust_threshold
        print(f"Hybrid Model initialized. Trusting N-gram when N >= {self.threshold}")

    def predict(self, context_str, first_letter):
        context_tokens = str(context_str).split()
        
        # 1. Try High-Order N-grams first
        # Traverse from Max_N down to Threshold
        for n in range(self.ngram_model.max_n, self.threshold - 1, -1):
            needed_context_len = n - 1
            if len(context_tokens) < needed_context_len:
                continue
            
            if needed_context_len > 0:
                current_context = tuple(context_tokens[-needed_context_len:])
            else:
                 current_context = ()

            candidates = self.ngram_model.models[n].get_best_candidates(current_context, first_letter)
            if candidates:
                # We found a match in a high-confidence N-gram layer. Return it.
                return candidates[0][0]

        # 2. If we reach here, N-gram failed or only has low-order (noisy) matches.
        # Use GPT-2.
        # We could also compare: if GPT-2 is confident, use it. If not, fallback to low-order N-gram.
        # For simplicity: Always trust GPT-2 over Bigram/Unigram.
        
        gpt_pred = self.gpt2_model.predict(context_str, first_letter)
        if gpt_pred:
            return gpt_pred
            
        # 3. Last resort: Low order N-gram (Bigram/Unigram)
        for n in range(self.threshold - 1, 0, -1):
             needed_context_len = n - 1
             if len(context_tokens) < needed_context_len:
                continue
             if needed_context_len > 0:
                current_context = tuple(context_tokens[-needed_context_len:])
             else:
                 current_context = ()
             
             candidates = self.ngram_model.models[n].get_best_candidates(current_context, first_letter)
             if candidates:
                 return candidates[0][0]
                 
        return ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit_train', type=int, default=1000000, help="Lines to train N-gram (default 1M)")
    parser.add_argument('--limit_dev', type=int, default=None)
    parser.add_argument('--max_n', type=int, default=5)
    parser.add_argument('--load_ngram', action='store_true', help="Load saved N-gram model instead of training")
    args = parser.parse_args()

    # 1. Load/Train N-Gram
    if args.load_ngram and os.path.exists(MODEL_SAVE_PATH):
        ngram = BackoffNGramModel.load(MODEL_SAVE_PATH)
    else:
        train_lines = load_training_data(TRAIN_PATH, limit=args.limit_train)
        print(f"Training {args.max_n}-gram model on {len(train_lines)} lines...")
        ngram = BackoffNGramModel(max_n=args.max_n)
        ngram.train(train_lines)
        os.makedirs(EXPERIMENT_DIR, exist_ok=True)
        ngram.save(MODEL_SAVE_PATH)

    # 2. Init GPT-2
    gpt2 = GPT2Predictor(model_name='gpt2')

    # 3. Create Hybrid
    # We trust 5-gram and 4-gram. If we only have 3-gram/2-gram matches, we ask GPT-2.
    hybrid = HybridModel(ngram, gpt2, ngram_trust_threshold=4)

    # 4. Evaluate
    dev_df = pd.read_csv(DEV_PATH)
    if args.limit_dev:
        dev_df = dev_df.head(args.limit_dev)
    
    print(f"Evaluating Hybrid Model on {len(dev_df)} examples...")
    accuracy = evaluate_model(hybrid, dev_df)

    log_experiment_result(
        EXPERIMENT_DIR, 
        'Hybrid (NGram + GPT2)', 
        f'N={args.max_n}, Trust>={hybrid.threshold}, Train={args.limit_train}', 
        args.limit_train, 
        accuracy
    )

    # 5. Generate Predictions
    # test_df = pd.read_csv(TEST_PATH_NO_ANSWER)
    # generate_test_predictions(hybrid, test_df, OUTPUT_PRED_PATH)

if __name__ == "__main__":
    main()
