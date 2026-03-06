import pandas as pd
import os
import numpy as np

def enrich_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    input_file = os.path.join(data_dir, 'train_split.csv')
    output_file = os.path.join(data_dir, 'train_split_enriched.csv')

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    # Analyze distribution
    # aspectCategory is a string representation of a list like "['food', 'price']" or just a string "food" depending on how it was saved.
    # The original csv had 'aspectCategory' as a single string per row.
    # contest1_train.csv has one row per aspect-sentiment pair.
    # train_split.csv was created from contest1_train.csv, so it has the same format.
    
    print("Original Distribution:")
    print(df['aspectCategory'].value_counts())
    
    # Oversampling Strategy
    # Target: 'price', 'ambience' (often minority classes)
    # We will duplicate rows with these aspects.
    
    minority_classes = ['price', 'ambience', 'conflict'] # conflict is polarity, but let's stick to aspect for now.
    
    # 1. Price: Oversample by 3x
    price_df = df[df['aspectCategory'] == 'price']
    
    # 2. Ambience: Oversample by 2x
    ambience_df = df[df['aspectCategory'] == 'ambience']
    
    # 3. Conflict Sentiment (hard to predict): Oversample by 2x
    conflict_df = df[df['polarity'] == 'conflict']
    
    enriched_df = pd.concat([df, price_df, price_df, ambience_df, conflict_df], ignore_index=True)
    
    # Shuffle
    enriched_df = enriched_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("\nEnriched Distribution:")
    print(enriched_df['aspectCategory'].value_counts())
    
    enriched_df.to_csv(output_file, index=False)
    print(f"\nSaved enriched training data to {output_file}")

if __name__ == "__main__":
    enrich_data()