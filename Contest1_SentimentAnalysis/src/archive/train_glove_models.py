import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
from utils import get_paths, save_and_evaluate, apply_heuristics
from model_glove import DAN, CNN, load_glove_embeddings

# Configuration
EMBED_DIM = 100
HIDDEN_DIM = 128
FILTER_SIZES = [3, 4, 5]
NUM_FILTERS = 100
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
MAX_LEN = 128

PATHS = get_paths()
TRAIN_FILE = os.path.join(PATHS['project_root'], 'data', 'train_split_enriched.csv')
DEV_FILE = os.path.join(PATHS['project_root'], 'data', 'dev_split.csv')
TEST_FILE = PATHS['test_csv']
GLOVE_PATH = os.path.join(PATHS['project_root'], 'data', 'glove.6B.100d.txt')

# --- Vocabulary Builder ---
def build_vocab(texts):
    counter = Counter()
    for text in texts:
        tokens = re.findall(r'\b\w+\b', text.lower())
        counter.update(tokens)
    
    # Keep words with count > 1 (simplification)
    vocab = {'<pad>': 0, '<unk>': 1}
    idx = 2
    for word, count in counter.most_common(20000): # Top 20k
        vocab[word] = idx
        idx += 1
    return vocab

def text_pipeline(text, vocab, max_len=MAX_LEN):
    tokens = re.findall(r'\b\w+\b', text.lower())
    indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
    # Padding / Truncation
    length = len(indices)
    if length > max_len:
        indices = indices[:max_len]
        length = max_len
    else:
        indices += [vocab['<pad>']] * (max_len - length)
    return torch.tensor(indices, dtype=torch.long), torch.tensor(max(1, length), dtype=torch.long)

class SimpleDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        x, length = text_pipeline(text, self.vocab)
        
        if self.labels is not None:
             # Handle scalar vs vector labels
             lbl = self.labels[idx]
             try:
                 # If list/array-like
                 _ = len(lbl) 
                 y = torch.tensor(lbl, dtype=torch.float)
             except TypeError:
                 # Scalar
                 y = torch.tensor(lbl, dtype=torch.long)
             return x, length, y
        else:
             return x, length

def train_model(model_cls, model_name, train_dl, val_dl, vocab, embedding_weights, num_classes, is_multilabel=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training {model_name} on {device}...")
    
    if model_name == 'DAN':
        model = DAN(len(vocab), EMBED_DIM, HIDDEN_DIM, num_classes, embedding_matrix=embedding_weights).to(device)
    else:
        model = CNN(len(vocab), EMBED_DIM, NUM_FILTERS, FILTER_SIZES, num_classes, embedding_matrix=embedding_weights).to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss() if is_multilabel else nn.CrossEntropyLoss()
    
    best_val_score = 0
    
    # Checkpoints
    ckpt_dir = os.path.join(PATHS['project_root'], 'checkpoints')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        
    save_path = os.path.join(ckpt_dir, f"best_{model_name}.pt")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, length, y in train_dl:
            x, length, y = x.to(device), length.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x, length)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x, length, y in val_dl:
                x, length, y = x.to(device), length.to(device), y.to(device)
                output = model(x, length)
                
                if is_multilabel:
                    preds = (torch.sigmoid(output) > 0.5).float()
                else:
                    preds = output.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        if is_multilabel:
            score = f1_score(all_labels, all_preds, average='micro')
        else:
            score = accuracy_score(all_labels, all_preds)
            
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}, Val Score = {score:.4f}")
        
        if score > best_val_score:
            best_val_score = score
            torch.save(model.state_dict(), save_path)
    
    # Load best
    model.load_state_dict(torch.load(save_path))
    return model

def main():
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(DEV_FILE)
    
    # Build Vocab
    vocab = build_vocab(train_df['text'].tolist() + val_df['text'].tolist())
    print(f"Vocab Size: {len(vocab)}")
    
    # Load Embeddings
    embeddings = load_glove_embeddings(GLOVE_PATH, vocab, EMBED_DIM)
    
    ASPECTS = ['ambience', 'anecdotes/miscellaneous', 'food', 'price', 'service']
    aspect2id = {label: i for i, label in enumerate(ASPECTS)}
    
    # --- Part 1: Aspect Detection (Multi-label) ---
    print("\n=== Training Aspect Detection (DAN & CNN) ===")
    
    # Prepare Labels
    def get_labels(df):
        all_texts = []
        all_labels = []
        grouped = df.groupby('id').agg({
            'text': 'first',
            'aspectCategory': lambda x: list(set(x))
        }).reset_index()
        
        for _, row in grouped.iterrows():
            lbl = [0] * len(ASPECTS)
            for asp in row['aspectCategory']:
                if asp in aspect2id:
                    lbl[aspect2id[asp]] = 1
            all_texts.append(row['text'])
            all_labels.append(lbl)
        return all_texts, np.array(all_labels)

    train_texts, train_y_asp = get_labels(train_df)
    val_texts, val_y_asp = get_labels(val_df)
    
    train_ds_asp = SimpleDataset(train_texts, train_y_asp, vocab)
    val_ds_asp = SimpleDataset(val_texts, val_y_asp, vocab)
    
    train_dl_asp = DataLoader(train_ds_asp, batch_size=BATCH_SIZE, shuffle=True)
    val_dl_asp = DataLoader(val_ds_asp, batch_size=BATCH_SIZE)
    
    # Train DAN
    model_dan_asp = train_model(DAN, 'DAN', train_dl_asp, val_dl_asp, vocab, embeddings, len(ASPECTS), is_multilabel=True)
    
    # Train CNN
    model_cnn_asp = train_model(CNN, 'CNN', train_dl_asp, val_dl_asp, vocab, embeddings, len(ASPECTS), is_multilabel=True)
    
    # --- Part 2: Sentiment Analysis (Multi-class) ---
    # Simplified: Predict sentiment given Text ONLY (Naïve approach for baseline comparison)
    # Ideally should concatenate Aspect + Text, but classical models struggle with that structure differently.
    # Let's just predict polarity of the text overall (or ignore aspect condition for this quick comparison).
    # Wait, the task is Aspect-Based Sentiment. 
    # To use GloVe models properly, we'd input "Aspect [SEP] Text".
    # Let's try appending Aspect name to text.
    
    print("\n=== Training Sentiment Analysis ===")
    POLARITIES = ['conflict', 'negative', 'neutral', 'positive']
    pol2id = {p: i for i, p in enumerate(POLARITIES)}
    
    def get_sent_data(df):
        texts = []
        labels = []
        for i, row in df.iterrows():
            # Input: "service the waiter was rude"
            texts.append(f"{row['aspectCategory']} {row['text']}")
            labels.append(pol2id[row['polarity']])
        return texts, np.array(labels)

    train_texts_sent, train_y_sent = get_sent_data(train_df)
    val_texts_sent, val_y_sent = get_sent_data(val_df)
    
    train_ds_sent = SimpleDataset(train_texts_sent, train_y_sent, vocab)
    val_ds_sent = SimpleDataset(val_texts_sent, val_y_sent, vocab)
    
    train_dl_sent = DataLoader(train_ds_sent, batch_size=BATCH_SIZE, shuffle=True)
    val_dl_sent = DataLoader(val_ds_sent, batch_size=BATCH_SIZE)
    
    # Train DAN Sentiment
    model_dan_sent = train_model(DAN, 'DAN_Sent', train_dl_sent, val_dl_sent, vocab, embeddings, len(POLARITIES), is_multilabel=False)
    
    # Train CNN Sentiment
    model_cnn_sent = train_model(CNN, 'CNN_Sent', train_dl_sent, val_dl_sent, vocab, embeddings, len(POLARITIES), is_multilabel=False)

    print("\n=== Generating Evaluation Reports ===")
    
    # Define Inference Function
    def run_inference(model_asp, model_sent, df, vocab, model_name, split_name):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_asp.eval()
        model_sent.eval()
        
        results = []
        texts = df['text'].tolist()
        ids = df['id'].tolist()
        
        # Maps
        id2aspect = {i: asp for i, asp in enumerate(ASPECTS)}
        id2polarity = {i: pol for i, pol in enumerate(POLARITIES)}
        
        print(f"[{model_name}] Predicting {len(df)} samples...")
        
        for i in range(len(df)):
            text = str(texts[i])
            review_id = ids[i]
            
            # 1. Predict Aspect
            x, length = text_pipeline(text, vocab)
            x = x.unsqueeze(0).to(device)
            length = length.unsqueeze(0).to(device) # Keep on CPU for index? No, model expects device
            
            with torch.no_grad():
                out_asp = model_asp(x, length)
                probs = torch.sigmoid(out_asp).cpu().numpy()[0]
                
            pred_indices = np.where(probs > 0.5)[0]
            pred_aspects = [id2aspect[idx] for idx in pred_indices]
            
            # Apply Heuristics
            pred_aspects = apply_heuristics(text, pred_aspects)
            
            # Fallback
            if not pred_aspects:
                max_idx = np.argmax(probs)
                pred_aspects = [id2aspect[max_idx]]
                pred_aspects = apply_heuristics(text, pred_aspects)
            
            # 2. Predict Sentiment for each aspect
            for aspect in pred_aspects:
                # Construct input as in training: "Aspect Text"
                input_text = f"{aspect} {text}"
                x_sent, len_sent = text_pipeline(input_text, vocab)
                x_sent = x_sent.unsqueeze(0).to(device)
                len_sent = len_sent.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    out_sent = model_sent(x_sent, len_sent)
                    sent_idx = out_sent.argmax(dim=1).item()
                
                results.append({
                    'id': review_id,
                    'aspectCategory': aspect,
                    'polarity': id2polarity[sent_idx]
                })
        
        return results

    # Run Inference on Dev Set
    # DAN
    print("\n--- Running DAN Inference (Dev) ---")
    dan_results = run_inference(model_dan_asp, model_dan_sent, val_df, vocab, "DAN", "validation")
    save_and_evaluate(dan_results, "val_preds_DAN.csv", "DAN_Baseline", "GloVe+DAN", "dev")
    
    # DAN Train
    print("\n--- Running DAN Inference (Train) ---")
    dan_train_results = run_inference(model_dan_asp, model_dan_sent, train_df, vocab, "DAN", "train")
    save_and_evaluate(dan_train_results, "train_preds_DAN.csv", "DAN_Baseline", "GloVe+DAN", "train")
    
    # CNN
    print("\n--- Running CNN Inference (Dev) ---")
    cnn_results = run_inference(model_cnn_asp, model_cnn_sent, val_df, vocab, "CNN", "validation")
    save_and_evaluate(cnn_results, "val_preds_CNN.csv", "CNN_Baseline", "GloVe+CNN", "dev")

    # CNN Train
    print("\n--- Running CNN Inference (Train) ---")
    cnn_train_results = run_inference(model_cnn_asp, model_cnn_sent, train_df, vocab, "CNN", "train")
    save_and_evaluate(cnn_train_results, "train_preds_CNN.csv", "CNN_Baseline", "GloVe+CNN", "train")

    print("\nDone! Evaluation reports saved to outputs/.")

if __name__ == "__main__":
    main()
