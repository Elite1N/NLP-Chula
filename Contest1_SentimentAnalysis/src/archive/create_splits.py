import pandas as pd
import os
from sklearn.model_selection import train_test_split

def create_splits():
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    train_file = os.path.join(data_dir, 'contest1_train.csv')
    
    # Check if exists
    if not os.path.exists(train_file):
        print(f"Error: {train_file} not found")
        return

    print(f"Reading {train_file}...")
    df = pd.read_csv(train_file)
    
    # Get unique IDs to ensure no leakage
    unique_ids = df['id'].unique()
    print(f"Total Unique Reviews: {len(unique_ids)}")
    
    # Split IDs (80/20 split)
    train_ids, dev_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)
    
    # Filter Dataframes
    train_df = df[df['id'].isin(train_ids)].copy()
    dev_df = df[df['id'].isin(dev_ids)].copy()
    
    # Save
    train_out = os.path.join(data_dir, 'train_split.csv')
    dev_out = os.path.join(data_dir, 'dev_split.csv')
    
    train_df.to_csv(train_out, index=False)
    dev_df.to_csv(dev_out, index=False)
    
    print(f"Created {train_out}: {len(train_df)} rows ({len(train_ids)} reviews)")
    print(f"Created {dev_out}: {len(dev_df)} rows ({len(dev_ids)} reviews)")

if __name__ == "__main__":
    create_splits()