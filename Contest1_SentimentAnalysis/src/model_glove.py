import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import re

def load_glove_embeddings(path, word2idx, embed_dim=100):
    if not os.path.exists(path):
        print(f"GloVe file not found at {path}. Using random embeddings.")
        return np.random.uniform(-0.1, 0.1, (len(word2idx), embed_dim)).astype(np.float32)
    
    print(f"Loading GloVe from {path}...")
    embeddings = np.random.uniform(-0.1, 0.1, (len(word2idx), embed_dim)).astype(np.float32)
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in word2idx:
                vector = np.array(values[1:], dtype='float32')
                embeddings[word2idx[word]] = vector
                
    return torch.tensor(embeddings)

class DAN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, embedding_matrix=None, dropout=0.3):
        super(DAN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)
            self.embedding.weight.requires_grad = False # Freeze GloVe? Or Fine-tune? Usually fine-tune or consistent.
            
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text, lengths):
        # text: [batch, seq_len]
        embedded = self.embedding(text) # [batch, seq_len, embed_dim]
        
        # Average pooling (ignoring padding ideally, but simplified here)
        # Sum over seq_len division by lengths for proper averaging
        # Here we use mean for simplicity on padded batches (pad=0 so it lowers mean, but acceptable for baseline)
        # Better: Sum then divide by actual lengths
        
        packed_embedded = torch.sum(embedded, dim=1) # [batch, embed_dim]
        # Divide by lengths to get mean
        lengths = lengths.unsqueeze(1).float()
        pooled = packed_embedded / lengths.clamp(min=1)
        
        x = F.relu(self.fc1(pooled))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes, output_dim, embedding_matrix=None, dropout=0.3):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)
            self.embedding.weight.requires_grad = False
            
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, 
                      out_channels=num_filters, 
                      kernel_size=fs)
            for fs in filter_sizes
        ])
        
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, lengths):
        # text: [batch, seq_len]
        embedded = self.embedding(text) # [batch, seq_len, embed_dim]
        embedded = embedded.permute(0, 2, 1) # [batch, embed_dim, seq_len]
        
        conved = [F.relu(conv(embedded)) for conv in self.convs] # [batch, num_filters, seq_len - filter_size + 1]
        
        # Max Pooling
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved] # [batch, num_filters]
        
        cat = torch.cat(pooled, dim=1)
        x = self.dropout(cat)
        return self.fc(x)
