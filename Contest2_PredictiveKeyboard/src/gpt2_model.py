import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging

class GPT2Predictor:
    def __init__(self, model_name='gpt2', device=None):
        print(f"Loading {model_name}...")
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Suppress huggingface warnings
        logging.getLogger("transformers").setLevel(logging.ERROR)

    def predict(self, context, first_letter):
        # GPT-2 was trained on natural text, but our input is PTB tokenized (lowercased, spaces, -lrb- etc).
        # Basic cleanup might help clean up the context slightly or we pass strictly.
        # Let's try to just pass the context as is for the baseline, maybe just ensuring it ends with space if needed?
        # Actually, GPT-2 next token prediction works best if we don't add trailing space if we want the next word immediately,
        # BUT here we are predicting a NEW word.
        
        # We need to filter candidates by first_letter.
        # Strategy: Get top-k logits, decode them, check first letter.
        
        input_ids = self.tokenizer.encode(context, return_tensors='pt').to(self.device)
        
        # optimization: if context is too long, truncate (GPT2 limit 1024 tokens)
        if input_ids.shape[1] > 1020:
            input_ids = input_ids[:, -1020:]
            
        with torch.no_grad():
            outputs = self.model(input_ids)
            predictions = outputs[0] # (batch, seq_len, vocab_size)
            
        # Get logits for the last token position
        next_token_logits = predictions[0, -1, :]
        
        # Sort by confidence
        # We need enough candidates to find one starting with 'first_letter'
        # k=1000 should be plenty
        k = 1000
        top_k_indices = torch.topk(next_token_logits, k).indices.tolist()
        
        for idx in top_k_indices:
            word = self.tokenizer.decode([idx]).strip().lower()
            
            # GPT-2 tokenizer often includes the leading space " word". .strip() removes it.
            # Check if valid word and starts with first_letter
            if len(word) > 0 and word.startswith(first_letter.lower()):
                # Clean word (sometimes tokens are partials or weird symbols)
                # For this task, we want simple words.
                # If word is just the letter itself, it counts.
                return word
                
        # Fallback if no match in top-k (rare)
        return ""
