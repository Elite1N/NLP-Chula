import pandas as pd
import os

def prepare_gold_data(file_path):
    """
    Reads the gold-standard romanization dataset.
    Extracts all possible valid romanizations for each Thai name.
    """
    df = pd.read_csv(file_path)
    
    samples = []
    # Drop rows with null names
    df = df.dropna(subset=['name'])
    
    for _, row in df.iterrows():
        thai_name = row['name'].strip()
        
        # Add up to 3 valid romanizations
        for r_col in ['romanize1', 'romanize2', 'romanize3']:
            if pd.notna(row[r_col]) and str(row[r_col]).strip() not in ['-', '—', '']:
                roman_name = str(row[r_col]).strip().lower()
                samples.append({'source': thai_name, 'target': roman_name})
                
    return pd.DataFrame(samples)

def prepare_silver_data(file_path, trust_threshold=0.8):
    """
    Reads the silver-standard dataset and filters it by a trust score.
    """
    df = pd.read_csv(file_path)
    
    # Filter by trust score if the column exists
    if 'trust score' in df.columns:
        df = df[df['trust score'] >= trust_threshold]
        
    samples = []
    df = df.dropna(subset=['name', 'romanize'])
    for _, row in df.iterrows():
        thai_name = row['name'].strip()
        roman_name = row['romanize'].strip().lower()
        samples.append({'source': thai_name, 'target': roman_name})
        
    return pd.DataFrame(samples)

if __name__ == "__main__":
    train_file = '../data/romanization-train.csv'
    silver_file = '../data/romanization-silver.csv'
    
    if os.path.exists(train_file):
        gold_df = prepare_gold_data(train_file)
        print(f"Generated {len(gold_df)} gold pairs.")
        gold_df.to_csv('../data/processed-train-gold.csv', index=False)
        
    if os.path.exists(silver_file):
        silver_df = prepare_silver_data(silver_file, trust_threshold=0.8) # Adjust threshold as needed
        print(f"Generated {len(silver_df)} silver pairs (threshold >= 0.8).")
        silver_df.to_csv('../data/processed-train-silver.csv', index=False)
