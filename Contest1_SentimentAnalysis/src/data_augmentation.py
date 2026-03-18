import pandas as pd
import os
import numpy as np
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

def get_back_translation_models():
    print("Loading translation models (EN <-> FR)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")
    
    # EN to FR
    model_name_en_fr = "Helsinki-NLP/opus-mt-en-fr"
    tokenizer_en_fr = MarianTokenizer.from_pretrained(model_name_en_fr)
    model_en_fr = MarianMTModel.from_pretrained(model_name_en_fr).to(device)
    
    # FR to EN
    model_name_fr_en = "Helsinki-NLP/opus-mt-fr-en"
    tokenizer_fr_en = MarianTokenizer.from_pretrained(model_name_fr_en)
    model_fr_en = MarianMTModel.from_pretrained(model_name_fr_en).to(device)
    
    return tokenizer_en_fr, model_en_fr, tokenizer_fr_en, model_fr_en, device

def back_translate(texts, tokenizer_en_fr, model_en_fr, tokenizer_fr_en, model_fr_en, device, batch_size=32):
    translated_final = []
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Back-Translating"):
        batch = texts[i:i+batch_size]
        
        # 1. EN -> FR
        inputs_ef = tokenizer_en_fr(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            translated_ef = model_en_fr.generate(**inputs_ef)
        texts_fr = tokenizer_en_fr.batch_decode(translated_ef, skip_special_tokens=True)
        
        # 2. FR -> EN
        inputs_fe = tokenizer_fr_en(texts_fr, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            translated_fe = model_fr_en.generate(**inputs_fe)
        texts_en_back = tokenizer_fr_en.batch_decode(translated_fe, skip_special_tokens=True)
        
        translated_final.extend(texts_en_back)
        
    return translated_final

def augment_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    input_file = os.path.join(data_dir, 'train_split.csv')
    output_file = os.path.join(data_dir, 'train_split_enriched.csv')

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    print("Original Distribution (Aspect):")
    print(df['aspectCategory'].value_counts())
    print("Original Distribution (Polarity):")
    print(df['polarity'].value_counts())
    
    # --- 1. Existing Oversampling (Duplicate Rows) ---
    # We identify minority/difficult subsets:
    # - Price (minority) -> Duplicate 3x
    # - Ambience (minority) -> Duplicate 2x
    # - Conflict (minority sentiment) -> Duplicate 2x
    
    # Selecting subsets
    price_df = df[df['aspectCategory'] == 'price']
    ambience_df = df[df['aspectCategory'] == 'ambience']
    conflict_df = df[df['polarity'] == 'conflict']
    
    oversampled_parts = [df] # Original data
    
    # Add duplications
    oversampled_parts.append(price_df) # +1x Price
    oversampled_parts.append(price_df) # +2x Price (Total 3x)
    oversampled_parts.append(ambience_df) # +1x Ambience (Total 2x)
    # oversampled_parts.append(conflict_df) # +1x Conflict (Total 2x) # We'll handle conflict separately via stronger back-translation or here? Let's stick to duplications.
   
    # Note: If conflict overlaps with price/ambience, it gets duplicated multiple times. That's fine.
    
    # --- 2. Back Translation Augmentation ---
    print("\nStarting Back Translation Augmentation...")
    
    # We select UNIQUE challenging samples to create paraphrased versions.
    # Target: 
    # - Low-count aspects: 'price', 'ambience'
    # - Difficult sentiment: 'conflict', 'neutral'
    # - Low-count intersection: 'anecdotes' + 'negative'? Maybe just stick to overall small classes.
    
    # Rule: Augment ALL 'price', 'ambience', 'conflict' samples via back-translation.
    # This acts as a smarter "duplicate" where the duplicate is slightly different.
    
    aug_subset = df[
        (df['aspectCategory'].isin(['price', 'ambience'])) | 
        (df['polarity'].isin(['conflict', 'neutral'])) 
    ].copy()
    
    # Remove duplicates from duplication logic above if any
    # Wait, df might contain duplicated IDs from original logic? No, train_split is unique rows.
    # But let's ensure we work on unique IDs.
    aug_subset = aug_subset.drop_duplicates(subset=['id'])
    
    print(f"Identified {len(aug_subset)} unique samples for Back-Translation...")
    
    if len(aug_subset) > 0:
        # Load heavy models only if needed
        tokenizer_en_fr, model_en_fr, tokenizer_fr_en, model_fr_en, device = get_back_translation_models()
        
        texts = aug_subset['text'].tolist()
        augmented_texts = back_translate(texts, tokenizer_en_fr, model_en_fr, tokenizer_fr_en, model_fr_en, device)
        
        aug_subset['text'] = augmented_texts
        aug_subset['id'] = aug_subset['id'].apply(lambda x: str(x) + "_aug") # Mark as augmented ID to avoid confusion
        
        oversampled_parts.append(aug_subset)
        print(f"Added {len(aug_subset)} back-translated samples.")
    
    # Combine everything
    enriched_df = pd.concat(oversampled_parts, ignore_index=True)
    
    # Shuffle
    enriched_df = enriched_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("\nFinal Enriched Distribution (Aspect):")
    print(enriched_df['aspectCategory'].value_counts())
    print("Final Enriched Distribution (Polarity):")
    print(enriched_df['polarity'].value_counts())
    
    enriched_df.to_csv(output_file, index=False)
    print(f"\nSaved enriched training data to {output_file}")

if __name__ == "__main__":
    augment_data()