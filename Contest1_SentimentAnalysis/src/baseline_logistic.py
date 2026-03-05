import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

def train_baseline():
    # Setup Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
    TRAIN_FILE = os.path.join(DATA_DIR, 'contest1_train.csv')
    TEST_FILE = os.path.join(DATA_DIR, 'contest1_test.csv')
    SUBMISSION_FILE = os.path.join(SCRIPT_DIR, '..', 'submission_baseline.csv')

    # Load Data
    print(f"Loading data from {TRAIN_FILE}...")
    if not os.path.exists(TRAIN_FILE):
        print(f"Error: {TRAIN_FILE} not found.")
        return

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    # Prepare Aspect Data (Multi-label)
    # Group by ID/Text to get all aspects for a single review
    grouped = train_df.groupby('id').agg({
        'text': 'first', 
        'aspectCategory': list,
        'polarity': list  # Note: Order matters, we assume aspectCategory and polarity lists correspond
    }).reset_index()

    # X = text, Y = aspects
    X = grouped['text']
    y_aspects = grouped['aspectCategory']

    # Binarize Aspects
    mlb = MultiLabelBinarizer()
    y_aspects_bin = mlb.fit_transform(y_aspects)
    classes = mlb.classes_
    print(f"Aspect Classes: {classes}")

    # Split for Validation
    X_train, X_val, y_train_aspect, y_val_aspect, ids_train, ids_val = train_test_split(
        X, y_aspects_bin, grouped['id'], test_size=0.2, random_state=42
    )

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
    
    unique_polarities = train_df['polarity'].unique()
    
    for aspect in classes:
        # Get rows for this aspect
        aspect_data = train_df[train_df['aspectCategory'] == aspect]
        
        # Filter train set for this aspect
        train_mask = aspect_data['id'].isin(ids_train)
        X_s_train = aspect_data[train_mask]['text']
        y_s_train = aspect_data[train_mask]['polarity']
        
        # Val set
        val_mask = aspect_data['id'].isin(ids_val)
        X_s_val = aspect_data[val_mask]['text']
        y_s_val = aspect_data[val_mask]['polarity']

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
            
    submission_df = pd.DataFrame(results)
    
    # Save
    print(f"Saving to {SUBMISSION_FILE}")
    submission_df.to_csv(SUBMISSION_FILE, index=False)
    print(submission_df.head())

if __name__ == "__main__":
    train_baseline()
