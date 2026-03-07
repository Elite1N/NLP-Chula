import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np

def create_stratified_splits():
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    train_file = os.path.join(data_dir, 'contest1_train.csv')
    
    if not os.path.exists(train_file):
        print(f"Error: {train_file} not found")
        return

    print(f"Reading {train_file}...")
    df = pd.read_csv(train_file)
    
    # helper to find rarity
    # Rarity order (approx based on counts): Price < Ambience < Service < Food < Misc
    rarity_map = {
        'price': 1,
        'ambience': 2,
        'service': 3,
        'food': 4,
        'anecdotes/miscellaneous': 5
    }
    
    # Group by ID to get aspects per review
    review_groups = df.groupby('id')['aspectCategory'].apply(list).reset_index()
    
    # Assign stratification label: The rarest aspect in the review
    def get_stratify_label(aspects):
        min_rank = 100
        for a in aspects:
            rank = rarity_map.get(a, 6)
            if rank < min_rank:
                min_rank = rank
        return min_rank

    review_groups['stratify_label'] = review_groups['aspectCategory'].apply(get_stratify_label)
    
    print("\nStratification Label Distribution (before split):")
    print(review_groups['stratify_label'].value_counts().sort_index())
    
    # Split IDs, stratifying on the label
    train_ids_df, dev_ids_df = train_test_split(
        review_groups, 
        test_size=0.2, 
        random_state=42, 
        stratify=review_groups['stratify_label']
    )
    
    train_ids = train_ids_df['id'].values
    dev_ids = dev_ids_df['id'].values
    
    # Filter Original Dataframes
    train_df = df[df['id'].isin(train_ids)].copy()
    dev_df = df[df['id'].isin(dev_ids)].copy()
    
    # Save
    train_out = os.path.join(data_dir, 'train_split.csv')
    dev_out = os.path.join(data_dir, 'dev_split.csv')
    
    train_df.to_csv(train_out, index=False)
    dev_df.to_csv(dev_out, index=False)
    
    print(f"\nCreated {train_out}: {len(train_df)} rows")
    print(f"Created {dev_out}: {len(dev_df)} rows")
    
    # Check new distribution
    print("\n--- New Balanced Distribution Check ---")
    for aspect in ['food', 'price', 'service', 'ambience', 'anecdotes/miscellaneous']:
        t_c = len(train_df[train_df['aspectCategory'] == aspect])
        d_c = len(dev_df[dev_df['aspectCategory'] == aspect])
        t_p = t_c / len(train_df) * 100
        d_p = d_c / len(dev_df) * 100
        print(f"{aspect:<25} | Train: {t_p:.1f}% ({t_c}) | Dev: {d_p:.1f}% ({d_c})")

if __name__ == "__main__":
    create_stratified_splits()