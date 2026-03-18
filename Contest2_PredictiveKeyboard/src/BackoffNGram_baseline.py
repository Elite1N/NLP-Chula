import os
import pandas as pd
from utils import load_training_data, evaluate_model, generate_test_predictions

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
    # For baseline, let's use a larger subset or full set if possible. 
    # Using 200k for now to match notebook, but you can increase this.
    train_lines = load_training_data(TRAIN_PATH, limit=200000)
    dev_df = pd.read_csv(DEV_PATH)
    test_df = pd.read_csv(TEST_PATH_NO_ANSWER)

    # 2. Train Model
    MAX_N = 3
    print(f"Initializing Backoff {MAX_N}-gram model...")
    model = BackoffNGramModel(max_n=MAX_N)
    model.train(train_lines)

    # 3. Evaluate
    accuracy = evaluate_model(model, dev_df)
    
    # Save Metrics
    with open(METRICS_PATH, 'w') as f:
        f.write(f"Model: Backoff {MAX_N}-gram\n")
        f.write(f"Training Data: 200000 lines\n")
        f.write(f"Dev Accuracy: {accuracy:.4f}\n")
    print(f"Metrics saved to {METRICS_PATH}")

    # 4. Generate Predictions for submission (if this was the final model)
    # create experiment dir if not exists
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    generate_test_predictions(model, test_df, OUTPUT_PRED_PATH)

if __name__ == "__main__":
    main()
