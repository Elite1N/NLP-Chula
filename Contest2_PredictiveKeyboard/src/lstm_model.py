import torch
import torch.nn as nn
from vocab import Vocabulary

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                            batch_first=True, dropout=(dropout if num_layers > 1 else 0))
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        # x: (batch, seq_len)
        embeds = self.dropout(self.embedding(x))
        # output: (batch, seq_len, hidden_dim)
        output, hidden = self.lstm(embeds, hidden)
        
        # We want predictions for the last step
        last_output = output[:, -1, :] 
        logits = self.fc(last_output)
        return logits, hidden

class LSTMPredictor:
    def __init__(self, model_path, vocab_path, embed_dim=128, hidden_dim=256, seq_len=5, device='cuda'):
        self.device = device
        self.seq_len = seq_len
        
        # Load vocab
        self.vocab = Vocabulary.load(vocab_path)
        
        # Load model
        self.model = LSTMModel(len(self.vocab), embed_dim, hidden_dim).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def predict(self, context, first_letter):
        # Tokenize (Numericalize)
        tokens = self.vocab.numericalize(context)
        
        # Handle length (Pad or Truncate)
        if len(tokens) < self.seq_len:
            # Pad from left with <PAD> (0)
            pads = [self.vocab.stoi["<PAD>"]] * (self.seq_len - len(tokens))
            tokens = pads + tokens
        else:
            # Truncate to last seq_len
            tokens = tokens[-self.seq_len:]
            
        x = torch.tensor([tokens]).to(self.device) # (1, seq_len)
        
        with torch.no_grad():
            logits, _ = self.model(x) # (1, vocab_size)
            probs = torch.softmax(logits, dim=1)
            
        # Get Top-K candidates to find match for first_letter
        k = 1000
        top_k_probs, top_k_indices = torch.topk(probs, k)
        
        top_k_indices = top_k_indices.squeeze().tolist()
        
        for idx in top_k_indices:
            word = self.vocab.itos[idx]
            
            # Skip special tokens
            if word in ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]:
                continue
                
            if word.lower().startswith(first_letter.lower()):
                return word
                
        return ""
