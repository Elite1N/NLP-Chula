import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Paths
TRAIN_FILE = 'data/contest1_train.csv'
TEST_FILE = 'data/contest1_test.csv'

def get_distribution(df, set_name):
    # Total Reviews (Unique IDs)
    num_reviews = df['id'].nunique()
    
    # Total Rows (Aspect-Sentiment pairs)
    num_rows = len(df)
    
    # Aspect Distribution
    aspect_counts = df['aspectCategory'].value_counts().to_dict()
    
    # Polarity Distribution
    polarity_counts = df['polarity'].value_counts().to_dict()
    
    return {
        'split': set_name,
        'reviews': num_reviews,
        'rows': num_rows,
        'aspects': aspect_counts,
        'polarities': polarity_counts
    }

def print_distribution(stats):
    print(f"\n--- {stats['split']} Set Statistics ---")
    print(f"Unique Reviews: {stats['reviews']}")
    print(f"Total Labels (Aspect-Polarity pairs): {stats['rows']}")
    
    print("\nAspect Distribution:")
    for aspect, count in stats['aspects'].items():
        print(f"  {aspect}: {count} ({count/stats['rows']:.1%})")
        
    print("\nPolarity Distribution:")
    if stats['polarities']:
        for polarity, count in stats['polarities'].items():
            print(f"  {polarity}: {count} ({count/stats['rows']:.1%})")
    else:
        print("  (No Polarity Labels)")

def main():
    if not os.path.exists(TRAIN_FILE):
        print("Train file not found.")
        return

    # Load Data
    full_train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    print("="*40)
    print("      DATASET DISTRIBUTION REPORT      ")
    print("="*40)

    # 1. Full Train and Test Stats
    print_distribution(get_distribution(full_train_df, "FULL TRAIN"))
    
    # Test set doesn't have labels, so just count reviews
    print(f"\n--- TEST Set Statistics ---")
    print(f"Unique Reviews: {test_df['id'].nunique()}")
    print(f"Total Rows: {len(test_df)}")
    print("(No labels in test set)")

    # 2. BERT Model Split Logic
    # Source: src/baseline_bert.py
    # Logic: Group by ID -> train_test_split(test_size=0.1, random_state=42)
    print("\n" + "="*40)
    print("      BERT MODEL SPLIT (10% Val)      ")
    print("="*40)
    
    unique_ids = full_train_df['id'].unique()
    train_ids_bert, val_ids_bert = train_test_split(unique_ids, test_size=0.1, random_state=42)
    
    bert_train_df = full_train_df[full_train_df['id'].isin(set(train_ids_bert))]
    bert_val_df = full_train_df[full_train_df['id'].isin(set(val_ids_bert))]
    
    print_distribution(get_distribution(bert_train_df, "BERT TRAIN"))
    print_distribution(get_distribution(bert_val_df, "BERT DEV"))

    # 3. Logistic Model Split Logic
    # Source: src/baseline_logistic.py
    # Logic: Group by ID -> train_test_split(test_size=0.2, random_state=42)
    print("\n" + "="*40)
    print("    LOGISTIC MODEL SPLIT (20% Val)    ")
    print("="*40)
    
    train_ids_log, val_ids_log = train_test_split(unique_ids, test_size=0.2, random_state=42)
    
    log_train_df = full_train_df[full_train_df['id'].isin(set(train_ids_log))]
    log_val_df = full_train_df[full_train_df['id'].isin(set(val_ids_log))]
    
    print_distribution(get_distribution(log_train_df, "LOGISTIC TRAIN"))
    print_distribution(get_distribution(log_val_df, "LOGISTIC DEV"))

    print("\n" + "="*40)
    print("           COMPARISON           ")
    print("="*40)
    overlap = len(set(val_ids_bert).intersection(set(val_ids_log)))
    print(f"Validation Set Overlap (Reviews): {overlap}")
    print(f"BERT Val Size: {len(val_ids_bert)}")
    print(f"Logistic Val Size: {len(val_ids_log)}")
    if set(val_ids_bert).issubset(set(val_ids_log)):
         print("Note: BERT Val set is a SUBSET of Logistic Val set (because random_state is same but size differs).")
    else:
         print("Note: Validation sets are different.")

if __name__ == "__main__":
    main()
