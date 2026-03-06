import pandas as pd
import numpy as np
import os
import sys

# Add src to path for utils import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import save_and_evaluate, save_submission, get_paths

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

def train_baseline():
    # Setup Paths
    paths = get_paths()
    TRAIN_SPLIT_FILE = os.path.join(paths['project_root'], 'data', 'train_split.csv')
    DEV_SPLIT_FILE = os.path.join(paths['project_root'], 'data', 'dev_split.csv')
    TEST_FILE = paths['test_csv']
    
    # Model Config
    MODEL_NAME = "Logistic Baseline"
    PARAMS = "C=1.0, tfidf=5000"

    # Load Data
    print(f"Loading data from splits...")
    if not os.path.exists(TRAIN_SPLIT_FILE) or not os.path.exists(DEV_SPLIT_FILE):
        print(f"Error: Splits not found. Run create_splits.py first.")
        return

    train_df = pd.read_csv(TRAIN_SPLIT_FILE)
    val_df = pd.read_csv(DEV_SPLIT_FILE)
    test_df = pd.read_csv(TEST_FILE)

    # Prepare Aspect Data (Multi-label)
    # Group by ID/Text to get all aspects for a single review
    grouped_train = train_df.groupby('id').agg({
        'text': 'first', 
        'aspectCategory': list
    }).reset_index()

    X_train = grouped_train['text']
    y_aspects_train = grouped_train['aspectCategory']
    ids_train = grouped_train['id']

    grouped_val = val_df.groupby('id').agg({
        'text': 'first', 
        'aspectCategory': list
    }).reset_index()

    X_val = grouped_val['text']
    y_aspects_val = grouped_val['aspectCategory']
    ids_val = grouped_val['id']

    # Binarize Aspects
    mlb = MultiLabelBinarizer()
    y_train_aspect = mlb.fit_transform(y_aspects_train)
    y_val_aspect = mlb.transform(y_aspects_val)
    classes = mlb.classes_
    print(f"Aspect Classes: {classes}")

    # train Aspect Model
    print("Training Aspect Model...")
    aspect_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))), # Improved TF-IDF
        ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear', C=1.0)))
    ])
    aspect_pipeline.fit(X_train, y_train_aspect)

    # Evaluate Aspect Model
    y_pred_aspect_val = aspect_pipeline.predict(X_val)
    print("\nAspect Classification Report:")
    print(classification_report(y_val_aspect, y_pred_aspect_val, target_names=classes))

    # Prepare Sentiment Data
    sentiment_models = {}
    print("Training Sentiment Models...")
    
    # We don't generally need unique polarities list for training logic below, but maybe later?
    unique_polarities = train_df['polarity'].unique()
    
    for aspect in classes:
        # Get rows for this aspect from TRAIN split
        aspect_data_train = train_df[train_df['aspectCategory'] == aspect]
        X_s_train = aspect_data_train['text']
        y_s_train = aspect_data_train['polarity']
        
        # Get rows for this aspect from VAL split
        aspect_data_val = val_df[val_df['aspectCategory'] == aspect]
        X_s_val = aspect_data_val['text']
        y_s_val = aspect_data_val['polarity']

        if len(X_s_train) < 5:
            print(f"Skipping {aspect} (not enough data: {len(X_s_train)})")
            continue

        sent_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', LogisticRegression(solver='lbfgs', max_iter=200)) # improved solver for multiclass
        ])
        sent_pipeline.fit(X_s_train, y_s_train)
        sentiment_models[aspect] = sent_pipeline
        
        if len(X_s_val) > 0:
            val_preds = sent_pipeline.predict(X_s_val)
            acc = accuracy_score(y_s_val, val_preds)
            print(f"Sentiment Acc for {aspect}: {acc:.4f} (Samples: {len(X_s_train)})")

    
    # Predict on Validation Set for Evaluation Logging
    print("\n--- Model Evaluation (Dev & Train) ---")

    # --- 1. EVALUATE ON DEV (Validation) ---
    print("\nPredicting on Validation Set...")
    # Use full X_val which is indexed by IDs in ids_val
    val_texts = X_val
    val_ids = ids_val
    
    val_aspects_bin = aspect_pipeline.predict(X_val)
    val_aspects_lists = mlb.inverse_transform(val_aspects_bin)
    
    val_results = []
    
    for text_id, text, aspects in zip(val_ids, val_texts, val_aspects_lists):
        final_aspects = list(aspects)
        if not final_aspects:
            final_aspects = ['anecdotes/miscellaneous']
            
        for aspect in final_aspects:
            if aspect in sentiment_models:
                sentiment = sentiment_models[aspect].predict([text])[0]
            else:
                sentiment = 'neutral' 
            
            val_results.append({'id': text_id, 'aspectCategory': aspect, 'polarity': sentiment})
            
    # Use standard Util function
    save_and_evaluate(val_results, 'val_preds_logistic.csv', MODEL_NAME, PARAMS, 'dev')

    # --- 2. EVALUATE ON TRAIN (Training Subset) ---
    print("\nPredicting on Training Subset...")
    # Use full X_train which is indexed by IDs in ids_train
    train_texts_subset = X_train
    train_ids_subset = ids_train
    
    train_aspects_bin = aspect_pipeline.predict(train_texts_subset)
    train_aspects_lists = mlb.inverse_transform(train_aspects_bin)
    
    train_results = []
    
    for text_id, text, aspects in zip(train_ids_subset, train_texts_subset, train_aspects_lists):
        final_aspects = list(aspects)
        if not final_aspects:
            final_aspects = ['anecdotes/miscellaneous']
            
        for aspect in final_aspects:
            if aspect in sentiment_models:
                sentiment = sentiment_models[aspect].predict([text])[0]
            else:
                sentiment = 'neutral' 
            
            train_results.append({'id': text_id, 'aspectCategory': aspect, 'polarity': sentiment})
            
    # Use standard Util function
    save_and_evaluate(train_results, 'train_preds_logistic.csv', MODEL_NAME, PARAMS, 'train')

    print("\nPredicting on Test Set...")
    # 1. Predict Aspects
    test_texts = test_df['text']
    test_ids = test_df['id']
    
    pred_aspects_bin = aspect_pipeline.predict(test_texts)
    pred_aspects_lists = mlb.inverse_transform(pred_aspects_bin)
    
    results = []
    
    for idx, (text_id, text, aspects) in enumerate(zip(test_ids, test_texts, pred_aspects_lists)):
        final_aspects = list(aspects)
        if not final_aspects:
            # Fallback
            final_aspects = ['anecdotes/miscellaneous']
            
        for aspect in final_aspects:
            if aspect in sentiment_models:
                sentiment = sentiment_models[aspect].predict([text])[0]
            else:
                sentiment = 'neutral' # Default fallback
            
            results.append({
                'id': text_id,
                'aspectCategory': aspect,
                'polarity': sentiment
            })
            
    # Save submission
    save_submission(results, 'submission_baseline.csv')

if __name__ == "__main__":
    train_baseline()
