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

        # Set pad token if it doesn't exist (GPT-2 doesn't have one by default)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # self.model.config.pad_token_id = self.model.config.eos_token_id # Optional but good practice

    def predict_batch(self, contexts, first_letters, batch_size=32):
        """
        Predicts next words for a batch of contexts and first_letters.
        Processing in batches is much faster on GPU.
        """
        results = []
        
        # Loop through data in chunks
        for i in range(0, len(contexts), batch_size):
            batch_contexts = contexts[i : i + batch_size]
            batch_first_letters = first_letters[i : i + batch_size]
            
            # Tokenize batch
            # padding=True ensures all sequences in batch are same length
            # truncation=True handles super long contexts
            inputs = self.tokenizer(batch_contexts, return_tensors='pt', padding=True, truncation=True, max_length=1024).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits # (batch, seq_len, vocab_size)

            # For each item in batch, get the last token's logits
            # Note: with padding, we need to be careful to take the logits of the last REAL token, 
            # or just the last token if we left padding to right.
            # Transformers tokenizer pads to the right by default? No, usually right text, left pad?
            # Actually GPT-2 is causal LM so right padding is tricky with attention mask. 
            # But for next token prediction, we just want the output at the last position of the input_ids (excluding padding if possible).
            # However, simpler approach: use left padding for generation, but here we just want logits.
            # If we pad right (default), the position of the last real token varies.
            
            # Let's check tokenizer padding side.
            # GPT-2 tokenizer defaults to right padding? Actually it has no pad token by default. We set it to EOS.
            
            # Better approach for batch inference effectively:
            # Just take the logits at the index of the last attention_mask=1 token.
            
            for j, (context, first_letter) in enumerate(zip(batch_contexts, batch_first_letters)):
                # Find length of this specific sequence (sum of attention mask)
                seq_len = inputs.attention_mask[j].sum().item()
                # Logits for the last real token are at index seq_len - 1
                next_token_logits = predictions[j, seq_len - 1, :]
                
                # Get top-k
                k = 1000
                top_k_indices = torch.topk(next_token_logits, k).indices.tolist()
                
                found_word = ""
                for idx in top_k_indices:
                    word = self.tokenizer.decode([idx]).strip().lower()
                    if len(word) > 0 and word.startswith(first_letter.lower()):
                        found_word = word
                        break
                results.append(found_word)
                
        return results

    def predict(self, context, first_letter):
        # Legacy single prediction (wraps batch)
        return self.predict_batch([context], [first_letter], batch_size=1)[0]
