import pandas as pd
import os

def check_split_distribution():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    train_file = os.path.join(data_dir, 'train_split.csv') # Original unenriched split
    dev_file = os.path.join(data_dir, 'dev_split.csv')
    
    if not os.path.exists(train_file):
        print("Train split not found (run create_splits.py first?)")
        return

    train_df = pd.read_csv(train_file)
    dev_df = pd.read_csv(dev_file)
    
    print("\n--- Class Distribution Check ---")
    print(f"Train Rows: {len(train_df)}")
    print(f"Dev Rows:   {len(dev_df)}")
    
    for aspect in ['food', 'price', 'service', 'ambience', 'anecdotes/miscellaneous']:
        train_count = len(train_df[train_df['aspectCategory'] == aspect])
        dev_count = len(dev_df[dev_df['aspectCategory'] == aspect])
        
        train_pct = train_count / len(train_df) * 100
        dev_pct = dev_count / len(dev_df) * 100
        
        print(f"{aspect:<25} | Train: {train_pct:.1f}% ({train_count}) | Dev: {dev_pct:.1f}% ({dev_count})")

if __name__ == "__main__":
    check_split_distribution()