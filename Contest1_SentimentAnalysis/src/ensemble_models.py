import pandas as pd
import numpy as np
import os
import sys
import argparse
import subprocess
from collections import Counter
from utils import get_paths, apply_heuristics

PATHS = get_paths()
EXPERIMENTS_DIR = os.path.join(PATHS['project_root'], 'experiments', 'v2')

# Define paths to model outputs (Validation set for tuning, Submission set for final)
VAL_FILES = {
    'deberta': os.path.join(PATHS['outputs_dir'], 'val_preds_deberta.csv'), # Updated to point to fresh output
    'roberta': os.path.join(EXPERIMENTS_DIR, 'roberta', 'val_preds_roberta.csv'),
    'distilbert': os.path.join(EXPERIMENTS_DIR, 'distilbert', 'val_preds_bert.csv')
}

SUB_FILES = {
    'deberta': os.path.join(PATHS['outputs_dir'], 'submission_deberta.csv'), # Updated to point to fresh output
    'roberta': os.path.join(EXPERIMENTS_DIR, 'roberta', 'submission_roberta.csv'),
    'distilbert': os.path.join(EXPERIMENTS_DIR, 'distilbert', 'submission_bert.csv')
}

def load_predictions(file_map):
    preds = {}
    valid_models = []
    
    for model_name, path in file_map.items():
        if os.path.exists(path):
            print(f"Loading {model_name} from {path}...")
            df = pd.read_csv(path)
            # Normalize column names just in case
            if 'aspectCategory' in df.columns:
                 df['aspectCategory'] = df['aspectCategory'].str.lower()
            preds[model_name] = df
            valid_models.append(model_name)
        else:
            print(f"Warning: {path} not found. Skipping {model_name}.")
            
    return preds, valid_models

def ensemble_predictions(preds_dict, models):
    if not models:
        return []

    # Get all unique IDs across all predictions
    all_ids = set()
    for m in models:
        all_ids.update(preds_dict[m]['id'].unique())
    
    all_ids = sorted(list(all_ids))
    ensemble_results = []
    
    print(f"Ensembling {len(models)} models for {len(all_ids)} reviews...")
    
    for review_id in all_ids:
        # 1. Collect predicted aspects
        model_aspects = {}
        all_predicted_aspects = []
        text = "" # Placeholder
        
        # Try to find text for heuristics
        for m in models:
            df = preds_dict[m]
            rows = df[df['id'] == review_id]
            if not rows.empty:
                if 'text' in rows.columns:
                     text = rows.iloc[0]['text']
                aspects = set(rows['aspectCategory'].values)
                model_aspects[m] = aspects
                all_predicted_aspects.extend(list(aspects))
        
        # Vote on Aspects
        # Threshold: Majority (2/3 -> 2, 2/2 -> 1 (Union))
        counts = Counter(all_predicted_aspects)
        threshold =  (len(models) + 1) // 2 
        
        if len(models) == 2:
            threshold = 1 # Union for 2 models
        
        final_aspects = [asp for asp, count in counts.items() if count >= threshold]
        
        # Heuristics Fallback
        if text:
             final_aspects = apply_heuristics(text, final_aspects)
        
        # Fallback if empty: most reliable model
        if not final_aspects:
             if 'deberta' in model_aspects and model_aspects['deberta']:
                 final_aspects = list(model_aspects['deberta'])
        
        # 2. Vote on Polarity
        for aspect in final_aspects:
            votes = []
            for m in models:
                df = preds_dict[m]
                row = df[(df['id'] == review_id) & (df['aspectCategory'] == aspect)]
                if not row.empty:
                    votes.append(row.iloc[0]['polarity'])
            
            if not votes:
                final_polarity = 'positive' # Default
            else:
                vote_counts = Counter(votes)
                # Tie breaking: DeBERTa > RoBERTa > Majority
                if len(vote_counts) > 1 and vote_counts.most_common(1)[0][1] == 1:
                     # Tie exists
                    if 'deberta' in models:
                        df = preds_dict['deberta']
                        row = df[(df['id'] == review_id) & (df['aspectCategory'] == aspect)]
                        if not row.empty:
                            final_polarity = row.iloc[0]['polarity']
                        else:
                            if 'roberta' in models: 
                                df = preds_dict['roberta']
                                row = df[(df['id'] == review_id) & (df['aspectCategory'] == aspect)]
                                if not row.empty:
                                    final_polarity = row.iloc[0]['polarity'] 
                                else:
                                     final_polarity = votes[0]
                    else:
                        final_polarity = votes[0]
                else:
                    final_polarity = vote_counts.most_common(1)[0][0]
            
            ensemble_results.append({
                'id': review_id,
                'aspectCategory': aspect,
                'polarity': final_polarity
            })
            
    return ensemble_results
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['submission', 'evaluate'], default='submission', help='Mode: submission or evaluate')
    args = parser.parse_args()
    
    print(f"=== Ensemble Model: {args.mode.upper()} ===")
    
    if args.mode == 'evaluate':
        files = VAL_FILES
        output_filename = 'val_preds_ensemble.csv'
    else:
        files = SUB_FILES
        output_filename = 'submission_ensemble.csv'
    
    preds, models = load_predictions(files)
    if not preds:
        print("Error: No predictions found.")
        return

    results = ensemble_predictions(preds, models)
    
    output_path = os.path.join(PATHS['outputs_dir'], output_filename)
    df = pd.DataFrame(results)
    
    # Ensure columns
    if not df.empty:
        df = df[['id', 'aspectCategory', 'polarity']]
    
    df.to_csv(output_path, index=False)
    print(f"Ensemble saved to {output_path}")
    
    if args.mode == 'evaluate':
        print("\nRunning Evaluation...")
        csv_log_path = os.path.join(PATHS['outputs_dir'], 'evaluation_Ensemble_log.csv')
        
        cmd = [
            sys.executable,
            PATHS['eval_script'],
            PATHS['train_csv'], 
            output_path,
            '--model', f'Ensemble ({"+".join(models)})',
            '--params', 'Majority Vote',
            '--split', 'dev',
            '--csv', csv_log_path
        ]
        subprocess.call(cmd)

if __name__ == "__main__":
    main()