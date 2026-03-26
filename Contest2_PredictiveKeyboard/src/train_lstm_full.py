import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import argparse
from tqdm import tqdm
from vocab import Vocabulary
from lstm_model import LSTMModel
import sys

# Increase recursion depth just in case, though not needed for iteration
sys.setrecursionlimit(2000)

class BigTextDataset(Dataset):
    def __init__(self, data_tensor, seq_len=5):
        """
        data_tensor: torch.LongTensor containing all token indices.
        """
        self.data_tensor = data_tensor
        self.seq_len = seq_len
        # Calculate length. We need at least seq_len + 1 tokens
        if len(data_tensor) <= seq_len:
            self.length = 0
        else:
            self.length = len(data_tensor) - seq_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Returns (x, y)
        # x is sequence of length seq_len
        # y is the next token
        # This slicing creates a new tensor, but it's small (seq_len).
        # data_tensor should be on CPU to avoid GPU OOM if dataset is huge.
        return self.data_tensor[idx : idx+self.seq_len], self.data_tensor[idx+self.seq_len]

def build_vocab_from_file(filepath, limit=None):
    print("Building vocabulary from full dataset...")
    vocab = Vocabulary(freq_threshold=3) 
    
    with open(filepath, 'r', encoding='utf-8') as f:
        # Create a generator for lines
        # Determine if limit applies
        if limit:
            lines_gen = (line for i, line in enumerate(f) if i < limit)
            vocab.build_vocabulary(lines_gen)
        else:
            vocab.build_vocabulary(f)
            
    return vocab

def load_and_tokenize(filepath, vocab, limit=None):
    """
    Reads file line by line, tokenizes, and returns a single 1D LongTensor of indices.
    """
    print(f"Tokenizing data from {filepath}...")
    token_indices = []
    
    # Pre-fetch special tokens
    sos_id = vocab.stoi["<SOS>"]
    eos_id = vocab.stoi["<EOS>"]
    unk_id = vocab.stoi["<UNK>"]
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in tqdm(enumerate(f)):
            if limit and i >= limit:
                break
            
            # Tokenize line
            # Manual tokenization to avoid split() overhead if possible, 
            # but vocab.numericalize uses split(), so we use it.
            # Using vocab.numericalize is fine for line-by-line.
            
            # Note: vocab.numericalize returns list of INTs.
            line_ids = vocab.numericalize(line)
            
            # We must flatten this.
            token_indices.append(sos_id)
            token_indices.extend(line_ids)
            token_indices.append(eos_id)
            
    print(f"Converting {len(token_indices)} tokens to tensor...")
    # Convert directly to LongTensor
    return torch.LongTensor(token_indices)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1) # 1 epoch for full dataset
    parser.add_argument('--batch_size', type=int, default=512) # Larger batch size
    parser.add_argument('--seq_len', type=int, default=5) 
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--limit', type=int, default=None) # Default None (all data)
    
    args = parser.parse_args()

    # Paths (Adjust for running from root)
    if os.path.exists('data/train.src.tok'):
        TRAIN_PATH = 'data/train.src.tok'
        EXPERIMENT_DIR = 'experiments/lstm_full' # Separate folder
    else:
        TRAIN_PATH = '../data/train.src.tok'
        EXPERIMENT_DIR = '../experiments/lstm_full'
    
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    
    # User requested name 'lstm_model_full' (implied .pth)
    MODEL_PATH = os.path.join(EXPERIMENT_DIR, 'lstm_model_full.pth')
    VOCAB_PATH = os.path.join(EXPERIMENT_DIR, 'vocab.pkl')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Build Vocab
    vocab = build_vocab_from_file(TRAIN_PATH, limit=args.limit)
    print(f"Vocab size: {len(vocab)}")
    vocab.save(VOCAB_PATH)

    # 2. Tokenize and Prepare Tensor
    data_tensor = load_and_tokenize(TRAIN_PATH, vocab, limit=args.limit)
    print(f"Total tokens: {len(data_tensor)}")

    # 3. Prepare Dataset
    dataset = BigTextDataset(data_tensor, seq_len=args.seq_len)
    # Using num_workers=0 to avoid multiprocessing overhead on Windows and simple tensor slicing logic
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True) 
    
    # 4. Init Model
    model = LSTMModel(len(vocab), args.embed_dim, args.hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Training Loop
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        batch_count = 0
        
        # tqdm wrapper
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for x, y in progress:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            progress.set_postfix(loss=loss.item())
            
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
