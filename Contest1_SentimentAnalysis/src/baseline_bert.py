import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import DataCollatorWithPadding

# Configuration
MODEL_NAME = "distilbert-base-uncased" # Faster than BERT, good for baseline
BATCH_SIZE = 16 
EPOCHS = 2
MAX_LEN = 128

# Setup Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
TRAIN_FILE = os.path.join(DATA_DIR, 'contest1_train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'contest1_test.csv')
SUBMISSION_FILE = os.path.join(SCRIPT_DIR, '..', 'submission_bert.csv')

# Labels
ASPECTS = ['ambience', 'anecdotes/miscellaneous', 'food', 'price', 'service']
POLARITIES = ['conflict', 'negative', 'neutral', 'positive']

aspect2id = {label: i for i, label in enumerate(ASPECTS)}
id2aspect = {i: label for label, i in aspect2id.items()}

polarity2id = {label: i for i, label in enumerate(POLARITIES)}
id2polarity = {i: label for label, i in polarity2id.items()}

class AspectDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len)
        item = {key: torch.tensor(val) for key, val in encoding.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

class SentimentDataset(Dataset):
    def __init__(self, texts, aspects, labels=None, tokenizer=None, max_len=MAX_LEN):
        self.texts = texts
        self.aspects = aspects
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        aspect = str(self.aspects[idx])
        # Format: [CLS] Aspect [SEP] Text [SEP]
        input_text = f"{aspect} [SEP] {text}" 
        
        encoding = self.tokenizer(input_text, truncation=True, padding="max_length", max_length=self.max_len)
        item = {key: torch.tensor(val) for key, val in encoding.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def compute_metrics_aspect(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.5).astype(int)
    # Micro F1 appropriate for multi-label Imbalance
    f1 = f1_score(labels, preds, average='micro')
    return {'f1_micro': f1}

def compute_metrics_sentiment(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

def main():
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # 1. Load Data
    if not os.path.exists(TRAIN_FILE):
        print(f"Error: {TRAIN_FILE} not found.")
        return
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    # ---------------------------------------------------------
    # PART A: Train Aspect Detection Model
    # ---------------------------------------------------------
    print("\n=== Training Aspect Detection Model ===")
    grouped = train_df.groupby('id').agg({
        'text': 'first',
        'aspectCategory': lambda x: list(set(x))
    }).reset_index()

    texts = grouped['text'].tolist()
    aspect_lists = grouped['aspectCategory'].tolist()
    
    # Multi-hot encoding
    labels = np.zeros((len(texts), len(ASPECTS)))
    for i, a_list in enumerate(aspect_lists):
        for aspect in a_list:
            if aspect in aspect2id:
                labels[i, aspect2id[aspect]] = 1

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset_aspect = AspectDataset(train_texts, train_labels, tokenizer)
    val_dataset_aspect = AspectDataset(val_texts, val_labels, tokenizer)

    model_aspect = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(ASPECTS),
        problem_type="multi_label_classification"
    )

    training_args_aspect = TrainingArguments(
        output_dir='./results_aspect',
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        logging_dir='./logs_aspect',
        logging_steps=50,
        learning_rate=2e-5,
        save_total_limit=1
    )

    trainer_aspect = Trainer(
        model=model_aspect,
        args=training_args_aspect,
        train_dataset=train_dataset_aspect,
        eval_dataset=val_dataset_aspect,
        compute_metrics=compute_metrics_aspect
    )

    trainer_aspect.train()
    
    # ---------------------------------------------------------
    # PART B: Train Sentiment Analysis Model
    # ---------------------------------------------------------
    print("\n=== Training Sentiment Analysis Model ===")
    # Prepare data: (text, aspect) -> polarity
    sent_train_texts = train_df['text'].tolist()
    sent_train_aspects = train_df['aspectCategory'].tolist()
    sent_train_labels = [polarity2id[p] for p in train_df['polarity'].tolist()]

    # Split
    s_train_texts, s_val_texts, s_train_aspects, s_val_aspects, s_train_y, s_val_y = train_test_split(
        sent_train_texts, sent_train_aspects, sent_train_labels, test_size=0.1, random_state=42
    )

    train_dataset_sent = SentimentDataset(s_train_texts, s_train_aspects, s_train_y, tokenizer)
    val_dataset_sent = SentimentDataset(s_val_texts, s_val_aspects, s_val_y, tokenizer)

    model_sent = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(POLARITIES)
    )

    training_args_sent = TrainingArguments(
        output_dir='./results_sentiment',
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir='./logs_sentiment',
        logging_steps=50,
        learning_rate=2e-5,
        save_total_limit=1
    )

    trainer_sent = Trainer(
        model=model_sent,
        args=training_args_sent,
        train_dataset=train_dataset_sent,
        eval_dataset=val_dataset_sent,
        compute_metrics=compute_metrics_sentiment
    )

    trainer_sent.train()

    # ---------------------------------------------------------
    # PART C: Inference on Test Set
    # ---------------------------------------------------------
    print("\n=== Predicting on Test Set ===")
    test_texts = test_df['text'].tolist()
    test_ids = test_df['id'].tolist()
    
    # 1. Predict Aspects
    test_dataset_aspect = AspectDataset(test_texts, None, tokenizer)
    aspect_preds_logits = trainer_aspect.predict(test_dataset_aspect).predictions
    aspect_probs = torch.sigmoid(torch.tensor(aspect_preds_logits)).numpy()
    
    results = []
    
    for i, (text_id, text) in enumerate(zip(test_ids, test_texts)):
        # Thresholding
        pred_indices = np.where(aspect_probs[i] > 0.5)[0]
        pred_aspects = [id2aspect[idx] for idx in pred_indices]
        
        # Fallback if no aspect predicted
        if not pred_aspects:
            # Use max probability class
            max_idx = np.argmax(aspect_probs[i])
            pred_aspects = [id2aspect[max_idx]]
        
        # 2. Predict Sentiment for each aspect
        # Prepare batch for sentiment
        # Input: [CLS] Aspect [SEP] Text
        batch_texts = [text] * len(pred_aspects)
        batch_aspects = pred_aspects
        
        # To batch predict, we can create a small dataset or just one-by-one (slow but fine for few)
        # Faster: Process all needed (text, aspect) pairs at once? 
        # For simplicity, let's just do individual inference or mini-batches per review
        # Or construct a flat list of all queries then map back.
        
        # Let's do simple flat list construction for faster inference
        for aspect in pred_aspects:
            results.append({
                'id': text_id,
                'text': text,
                'aspect': aspect,
                'orig_index': i
            })

    # Prepare sentiment inference dataset
    inf_texts = [r['text'] for r in results]
    inf_aspects = [r['aspect'] for r in results]
    
    inf_dataset = SentimentDataset(inf_texts, inf_aspects, None, tokenizer)
    sent_preds_logits = trainer_sent.predict(inf_dataset).predictions
    sent_preds_ids = np.argmax(sent_preds_logits, axis=-1)
    
    final_output = []
    for i, r in enumerate(results):
        polarity = id2polarity[sent_preds_ids[i]]
        final_output.append({
            'id': r['id'],
            'aspectCategory': r['aspect'],
            'polarity': polarity
        })
    
    # ---------------------------------------------------------
    # PART D: Prediction on Validation Set (for logging)
    # ---------------------------------------------------------
    print("\n=== Predicting on Validation Set (for logging) ===")
    # We used train_test_split on 'texts' earlier, but we need the original IDs to map back to Gold.
    # The split was:
    # train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, ...)
    # This split was on the UNIQUE texts from grouped dataframe.
    # We need to reconstruct which IDs belong to val set.
    
    # Re-run the split logic to get the indices/IDs
    # grouped = train_df.groupby('id')...
    # texts = grouped['text'].tolist()
    # IDs = grouped['id'].tolist()
    
    grouped_ids = grouped['id'].tolist()
    train_ids_split, val_ids_split, _, _ = train_test_split(
        grouped_ids, labels, test_size=0.1, random_state=42
    )

    # Now we have the IDs in the validation set: val_ids_split
    # We need to get the texts for these IDs
    # We can filter the original dataframe or use the grouped texts
    
    val_id_set = set(val_ids_split)
    
    # Filter original df to get (id, text) pairs for these validation IDs
    # Note: A single ID has one text, but multiple aspects.
    val_rows_df = train_df[train_df['id'].isin(val_id_set)].groupby('id').first().reset_index()
    val_eval_ids = val_rows_df['id'].tolist()
    val_eval_texts = val_rows_df['text'].tolist()

    # 1. Predict Aspects on Val
    val_dataset_aspect = AspectDataset(val_eval_texts, None, tokenizer)
    val_aspect_logits = trainer_aspect.predict(val_dataset_aspect).predictions
    val_aspect_probs = torch.sigmoid(torch.tensor(val_aspect_logits)).numpy()
    
    val_results_rows = []
    
    for i, (text_id, text) in enumerate(zip(val_eval_ids, val_eval_texts)):
        pred_indices = np.where(val_aspect_probs[i] > 0.5)[0]
        pred_aspects = [id2aspect[idx] for idx in pred_indices]
        
        if not pred_aspects:
            max_idx = np.argmax(val_aspect_probs[i])
            pred_aspects = [id2aspect[max_idx]]
            
        for aspect in pred_aspects:
            val_results_rows.append({
                'id': text_id,
                'text': text,
                'aspect': aspect
            })
            
    # 2. Predict Sentiment on Val
    if val_results_rows:
        val_inf_texts = [r['text'] for r in val_results_rows]
        val_inf_aspects = [r['aspect'] for r in val_results_rows]
        
        val_inf_dataset = SentimentDataset(val_inf_texts, val_inf_aspects, None, tokenizer)
        val_sent_logits = trainer_sent.predict(val_inf_dataset).predictions
        val_sent_ids = np.argmax(val_sent_logits, axis=-1)
        
        final_val_output = []
        for i, r in enumerate(val_results_rows):
            polarity = id2polarity[val_sent_ids[i]]
            final_val_output.append({
                'id': r['id'],
                'aspectCategory': r['aspect'],
                'polarity': polarity
            })
            
        val_preds_file = os.path.join(SCRIPT_DIR, '..', 'val_preds_bert.csv')
        pd.DataFrame(final_val_output).to_csv(val_preds_file, index=False)
        print(f"Validation predictions saved to {val_preds_file}")
        
        # Evaluate
        import sys
        import subprocess
        print("Running evaluation on DEV...")
        eval_script = os.path.join(SCRIPT_DIR, "evaluate.py")
        subprocess.call([
            sys.executable,
            eval_script,
            TRAIN_FILE,
            val_preds_file,
            '--model', 'DistilBERT Baseline',
            '--params', f'Ep={EPOCHS}, LR=2e-5',
            '--split', 'dev'
        ])
        
    # ---------------------------------------------------------
    # PART E: Prediction on Training Set (for logging)
    # ---------------------------------------------------------
    print("\n=== Predicting on Training Subset (for logging) ===")
    
    # Reconstruct training IDs and texts
    # train_ids_split contains the training IDs
    train_id_set = set(train_ids_split)
    
    train_rows_df = train_df[train_df['id'].isin(train_id_set)].groupby('id').first().reset_index()
    train_eval_ids = train_rows_df['id'].tolist()
    train_eval_texts = train_rows_df['text'].tolist()

    # 1. Predict Aspects on Train
    train_dataset_aspect = AspectDataset(train_eval_texts, None, tokenizer)
    train_aspect_logits = trainer_aspect.predict(train_dataset_aspect).predictions
    train_aspect_probs = torch.sigmoid(torch.tensor(train_aspect_logits)).numpy()
    
    train_results_rows = []
    
    for i, (text_id, text) in enumerate(zip(train_eval_ids, train_eval_texts)):
        pred_indices = np.where(train_aspect_probs[i] > 0.5)[0] # Threshold
        pred_aspects = [id2aspect[idx] for idx in pred_indices]
        
        if not pred_aspects:
            max_idx = np.argmax(train_aspect_probs[i])
            pred_aspects = [id2aspect[max_idx]]
            
        for aspect in pred_aspects:
            train_results_rows.append({
                'id': text_id,
                'text': text,
                'aspect': aspect
            })
            
    # 2. Predict Sentiment on Train
    if train_results_rows:
        train_inf_texts = [r['text'] for r in train_results_rows]
        train_inf_aspects = [r['aspect'] for r in train_results_rows]
        
        train_inf_dataset = SentimentDataset(train_inf_texts, train_inf_aspects, None, tokenizer)
        train_sent_logits = trainer_sent.predict(train_inf_dataset).predictions
        train_sent_ids = np.argmax(train_sent_logits, axis=-1)
        
        final_train_output = []
        for i, r in enumerate(train_results_rows):
            polarity = id2polarity[train_sent_ids[i]]
            final_train_output.append({
                'id': r['id'],
                'aspectCategory': r['aspect'],
                'polarity': polarity
            })
            
        train_preds_file = os.path.join(SCRIPT_DIR, '..', 'train_preds_bert.csv')
        pd.DataFrame(final_train_output).to_csv(train_preds_file, index=False)
        print(f"Training predictions saved to {train_preds_file}")
        
        # Evaluate
        print("Running evaluation on TRAIN...")
        subprocess.call([
            sys.executable,
            eval_script,
            TRAIN_FILE,
            train_preds_file,
            '--model', 'DistilBERT Baseline',
            '--params', f'Ep={EPOCHS}, LR=2e-5',
            '--split', 'train'
        ])
    
    # Save
    submission_df = pd.DataFrame(final_output)
    submission_df.to_csv(SUBMISSION_FILE, index=False)
    print(f"Saved submission to {SUBMISSION_FILE}")
    print(submission_df.head())

if __name__ == "__main__":
    main()
