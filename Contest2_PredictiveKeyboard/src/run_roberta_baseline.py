import os
import torch
import argparse
import pandas as pd
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm
from utils import load_training_data, log_experiment_result, generate_test_predictions, evaluate_model

# -----------------
# Predictor Class
# -----------------
class RobertaPredictor:
    def __init__(self, model_name_or_path, device='cpu'):
        print(f"Loading RoBERTa model from {model_name_or_path}...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path).to(device)
        self.model.eval()
        
    def predict(self, context, first_letter):
        # 1. Prepare Input: "Context <mask_token>"
        # RoBERTa's mask token is <mask>
        text = f"{context} {self.tokenizer.mask_token}"
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        
        # 2. Forward Pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits
            
        # 3. Get Mask Token Logits
        # Find the index of the mask token
        mask_token_index = (inputs.input_ids == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        
        # Get logits for the mask token
        mask_token_logits = predictions[0, mask_token_index, :]
        
        # 4. Get Top K Candidates
        # We need to find words that start with first_letter
        # Since RoBERTa uses BPE, tokens might be partial words.
        # We want tokens that start with the letter.
        # However, RoBERTa tokenizer treats space as specialized char (Ġ).
        # " word" -> Ġword.
        # If context ends with space, next token likely starts with Ġ.
        
        # Let's get top 1000 candidates and filter
        top_k = 1000
        probs = torch.softmax(mask_token_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k)
        
        top_indices = top_indices.squeeze().tolist()
        
        for token_id in top_indices:
            token_str = self.tokenizer.decode([token_id]).strip()
            
            # Filter
            # 1. Must start with first_letter (case insensitive)
            if not token_str.lower().startswith(first_letter.lower()):
                continue
                
            # 2. Must not be just the letter itself (unless the answer is single letter "I" or "a")
            # Usually we want a word.
            # Also RoBERTa might predict "<s>" or "</s>" or other special tokens?
            if token_str.lower() in [self.tokenizer.bos_token, self.tokenizer.eos_token, self.tokenizer.pad_token, self.tokenizer.mask_token, self.tokenizer.unk_token]:
                continue
                
            # 3. Clean up special chars if any (RoBERTa uses Ġ)
            # The decode method usually handles Ġ -> space, but we stripped.
            
            return token_str
            
        # Fallback if nothing found (should be rare with top_k=1000)
        return ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='experiments/roberta_baseline')
    parser.add_argument('--limit_dev', type=int, default=None)
    parser.add_argument('--generate_submission', action='store_true', default=True)
    args = parser.parse_args()

    # Paths
    if not os.path.exists(args.model_path):
        if os.path.exists('../' + args.model_path):
            args.model_path = '../' + args.model_path
            DEV_PATH = '../data/dev_set.csv'
            TEST_PATH = '../data/test_set_no_answer.csv'
        else:
             DEV_PATH = 'data/dev_set.csv'
             TEST_PATH = 'data/test_set_no_answer.csv'
    else:
        DEV_PATH = 'data/dev_set.csv'
        TEST_PATH = 'data/test_set_no_answer.csv'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Predictor
    try:
        predictor = RobertaPredictor(args.model_path, device=str(device))
    except Exception as e:
        print(f"Error loading model from {args.model_path}: {e}")
        print("Falling back to 'distilroberta-base' (untrained) to demonstrate...")
        predictor = RobertaPredictor('distilroberta-base', device=str(device))

    # 2. Evaluate
    print(f"Loading Dev Set from {DEV_PATH}...")
    if os.path.exists(DEV_PATH):
        dev_df = pd.read_csv(DEV_PATH)
        if args.limit_dev:
            dev_df = dev_df.head(args.limit_dev)
            
        print(f"Evaluating RoBERTa on {len(dev_df)} examples...")
        accuracy = evaluate_model(predictor, dev_df)
        print(f"Accuracy: {accuracy}")
        
        # Save Metrics
        log_experiment_result(
            args.model_path, 
            'RoBERTa Baseline', 
            'Model=distilroberta-base, MaskedLM', 
            '100k lines (Subset)', 
            accuracy,
            filename='evaluations.csv'
        )
    else:
        print(f"Dev set not found at {DEV_PATH}")

    # 3. Generate Submission
    if args.generate_submission:
        if os.path.exists(TEST_PATH):
            print(f"Generating predictions for {TEST_PATH}...")
            test_df = pd.read_csv(TEST_PATH)
            output_path = os.path.join(args.model_path, 'test_set_pred_roberta.txt')
            generate_test_predictions(predictor, test_df, output_path)
            print(f"Saved predictions to {output_path}")
        else:
            print(f"Test set not found at {TEST_PATH}")

if __name__ == "__main__":
    main()
