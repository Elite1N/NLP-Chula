import pandas as pd
import numpy as np
import os
from collections import Counter, defaultdict
import time
import re
import csv
from utils import load_training_data, evaluate_model, generate_test_predictions, log_experiment_result

# Conf
TRAIN_PATH = '../data/train.src.tok'
DEV_PATH = '../data/dev_set.csv'
TEST_PATH_NO_ANSWER = '../data/test_set_no_answer.csv'
EXPERIMENT_DIR = '../experiments/ngram_baseline'
OUTPUT_PRED_PATH = os.path.join(EXPERIMENT_DIR, 'test_set_pred.txt')
METRICS_PATH = os.path.join(EXPERIMENT_DIR, 'metrics.txt')
class NGramModel:
    def __init__(self, n=3):
        self.n = n
        self.counts = defaultdict(Counter)
        self.context_counts = Counter() # Count of times a context appears (for probability calc if needed, though mostly we just need max count here)
    
    def train(self, lines):
        """
        Trains the N-gram model on the list of tokenized strings (lines).
        """
        print(f"Training {self.n}-gram model...")
        start_time = time.time()
        for line in lines: # Iterating through lines (sentences)
            tokens = line.split() # The data is already tokenized with spaces
            
            # Skip if sentence is shorter than n
            if len(tokens) < self.n:
                continue

            # Sliding window to get n-grams
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i : i + self.n - 1])
                word = tokens[i + self.n - 1]
                self.counts[context][word] += 1
                self.context_counts[context] += 1
                
        print(f"Training finished in {time.time() - start_time:.2f} seconds.")
        print(f"Number of unique contexts: {len(self.counts)}")

    def get_count(self, context):
        return self.counts.get(tuple(context), Counter())

    def get_best_candidates(self, context, first_letter):
        """
        Returns list of (word, count) candidates starting with first_letter
        given the context tuple.
        """
        candidates = self.counts.get(tuple(context), Counter())
        # Filter by first letter
        filtered = [(word, count) for word, count in candidates.items() if str(word).lower().startswith(first_letter.lower())]
        # Sort by count desc
        filtered.sort(key=lambda x: x[1], reverse=True)
        return filtered

    def predict(self, context_str, first_letter):
        context_tokens = str(context_str).split()
        needed_len = self.n - 1
        
        if needed_len > 0:
            if len(context_tokens) < needed_len:
                return ""
            context = tuple(context_tokens[-needed_len:])
        else:
            context = ()
            
        candidates = self.get_best_candidates(context, first_letter)
        if candidates:
            return candidates[0][0]
        return ""

class BackoffNGramModel:
    def __init__(self, max_n=5):
        self.models = {}
        self.max_n = max_n
        for n in range(1, max_n + 1):
            self.models[n] = NGramModel(n)

    def train(self, lines):
        for n in range(1, self.max_n + 1):
            self.models[n].train(lines)

    def predict(self, context_str, first_letter):
        # Tokenize context just in case (though input is string)
        # The context in csv is a string. "word1 word2 word3 ..."
        context_tokens = str(context_str).split()
        
        # Try from max_n down to 1
        for n in range(self.max_n, 1, -1):
            # For N-gram prediction, we need the last N-1 words of context
            needed_context_len = n - 1
            
            # If context is too short for this N, skip to lower N
            if len(context_tokens) < needed_context_len:
                continue
            
            # Get the last N-1 tokens as context
            if needed_context_len > 0:
                current_context = tuple(context_tokens[-needed_context_len:])
            else:
                 current_context = () # Should handle bigram context being 1 word, wait n=2 -> context len 1. n=1 -> context len 0.
            
            candidates = self.models[n].get_best_candidates(current_context, first_letter)
            
            if candidates:
                return candidates[0][0] # Return best match
        
        # If still no match, look at unigram (1-gram) model
        # 1-gram model (N=1) context is empty tuple () 
        # But wait, 1-gram usually implies P(w). My NGramModel implementation for n=1:
        # loop range(len - 0): context=tokens[i:i], word=tokens[i]. context is indeed ().
        candidates = self.models[1].get_best_candidates((), first_letter)
        if candidates:
            return candidates[0][0]
            
        return "" # Should not happen ideally if unigram covers all vocab check

def main():
    # 1. Load Data
    # Using 200k at first, might experiment on larger training samples
    train_limit = 200000
    train_lines = load_training_data(TRAIN_PATH, limit=train_limit)
    dev_df = pd.read_csv(DEV_PATH)
    test_df = pd.read_csv(TEST_PATH_NO_ANSWER)

    # 2. Train Model
    MAX_N = 3  #TODO: change this to change the N-gram config!
    print(f"Initializing Backoff {MAX_N}-gram model...")
    # Training the backoff model trains all n-gram models from 1 to MAX_N
    backoff_model = BackoffNGramModel(max_n=MAX_N)
    backoff_model.train(train_lines)

    # 3. Evaluate Comparisons
    print("\n--- Evaluating Trigram (No Backoff) ---")
    # Access the specific NGramModel for n=MAX_N layer from the backoff collection
    trigram_no_backoff = backoff_model.models[MAX_N]
    acc_no_backoff = evaluate_model(trigram_no_backoff, dev_df)

    print("\n--- Evaluating Trigram (With Backoff) ---")
    acc_backoff = evaluate_model(backoff_model, dev_df)
    
    # Save Metrics to CSV
    log_experiment_result(
        EXPERIMENT_DIR, 
        'N-Gram (No Backoff) Model', 
        f'N={MAX_N}', 
        train_limit, 
        acc_no_backoff
    )
    log_experiment_result(
        EXPERIMENT_DIR, 
        'Backoff N-Gram Model', 
        f'Max N={MAX_N}', 
        train_limit, 
        acc_backoff
    )

    # 4. Generate Predictions for submission (using the better model, usually backoff)
    OUTPUT_PRED_PATH = os.path.join(EXPERIMENT_DIR, f'test_set_pred_{MAX_N}_{train_limit}.txt') #TODO: this is bandaid, might do sumthing later
    generate_test_predictions(backoff_model, test_df, OUTPUT_PRED_PATH )

if __name__ == "__main__":
    main()
