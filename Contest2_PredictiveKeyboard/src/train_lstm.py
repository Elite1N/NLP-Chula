import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import argparse
from tqdm import tqdm
from vocab import Vocabulary
from lstm_model import LSTMModel
from utils import load_training_data

class TextDataset(Dataset):
    def __init__(self, lines, vocab, seq_len=5):
        self.vocab = vocab
        self.seq_len = seq_len
        self.data = []
        
        print("Preparing dataset (sliding windows)...")
        for line in tqdm(lines):
            # Tokenize
            tokens = [vocab.stoi["<SOS>"]] + vocab.numericalize(line) + [vocab.stoi["<EOS>"]]
            
            if len(tokens) <= seq_len:
                continue
                
            # Sliding window: Input = tokens[i:i+seq_len], Target = tokens[i+seq_len]
            # Actually standard LM predicts next token for every step. 
            # But for simplicity in this baseline, let's do Many-to-One.
            # Context window -> Next Word.
            
            for i in range(len(tokens) - seq_len):
                x = tokens[i:i+seq_len]
                y = tokens[i+seq_len]
                self.data.append((torch.tensor(x), torch.tensor(y)))
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seq_len', type=int, default=5) # Context window size
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--limit', type=int, default=50000) # Small subset for speed
    args = parser.parse_args()

    # Paths (Adjust for running from root)
    if os.path.exists('data/train.src.tok'):
        TRAIN_PATH = 'data/train.src.tok'
        EXPERIMENT_DIR = 'experiments/lstm_baseline'
    else:
        TRAIN_PATH = '../data/train.src.tok'
        EXPERIMENT_DIR = '../experiments/lstm_baseline'
    
    MODEL_PATH = os.path.join(EXPERIMENT_DIR, 'lstm_model.pth')
    VOCAB_PATH = os.path.join(EXPERIMENT_DIR, 'vocab.pkl')
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    lines = load_training_data(TRAIN_PATH, limit=args.limit)
    
    # 2. Build Vocab
    print("Building vocabulary...")
    vocab = Vocabulary(freq_threshold=2)
    vocab.build_vocabulary(lines)
    print(f"Vocab size: {len(vocab)}")
    vocab.save(VOCAB_PATH)

    # 3. Prepare Dataset
    dataset = TextDataset(lines, vocab, seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 4. Init Model
    model = LSTMModel(len(vocab), args.embed_dim, args.hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Training Loop
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for x, y in progress:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())
            
        print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
