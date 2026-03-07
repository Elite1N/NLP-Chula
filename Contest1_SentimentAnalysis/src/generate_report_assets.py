import pandas as pd
import numpy as np
import os
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTS_AVAILABLE = True
except ImportError:
    PLOTS_AVAILABLE = False
    print("Warning: Matplotlib or Seaborn not found. Plots will be skipped.")

from sklearn.metrics import confusion_matrix, classification_report


# Paths - Updated based on file search
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) # src/
DATA_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), 'data')
EXPERIMENTS_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), 'experiments')

# Specific paths found
GT_FILE = os.path.join(DATA_DIR, 'dev_split.csv')
DEBERTA_PREDS = os.path.join(EXPERIMENTS_DIR, 'v2', 'DeBERTa', 'val_preds_deberta.csv')
ROBERTA_PREDS = os.path.join(EXPERIMENTS_DIR, 'v2', 'roberta', 'val_preds_roberta.csv')
OUTPUT_REPORT_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), 'report_assets')

if not os.path.exists(OUTPUT_REPORT_DIR):
    os.makedirs(OUTPUT_REPORT_DIR)

def load_data():
    print("Loading data...")
    gt_df = pd.read_csv(GT_FILE)
    deberta_df = pd.read_csv(DEBERTA_PREDS)
    roberta_df = pd.read_csv(ROBERTA_PREDS)
    
    # Ensure all aspects are lowercase just in case
    gt_df['aspectCategory'] = gt_df['aspectCategory'].str.lower()
    deberta_df['aspectCategory'] = deberta_df['aspectCategory'].str.lower()
    roberta_df['aspectCategory'] = roberta_df['aspectCategory'].str.lower()
    
    return gt_df, deberta_df, roberta_df

def analyze_model(model_name, pred_df, gt_df):
    print(f"\nAnalyzing {model_name}...")
    
    # Merge on ID and Aspect
    merged = pd.merge(
        gt_df, 
        pred_df, 
        on=['id', 'aspectCategory'], 
        how='outer', 
        suffixes=('_gt', '_pred'),
        indicator=True
    )
    
    # Identify Types of errors
    # 1. False Positives (In Pred, Not in GT) -> Right_only
    fp = merged[merged['_merge'] == 'right_only'].copy()
    
    # 2. False Negatives (In GT, Not in Pred) -> Left_only
    fn = merged[merged['_merge'] == 'left_only'].copy()
    
    # 3. True Positives (Aspect Correct), check Polarity
    tp = merged[merged['_merge'] == 'both'].copy()
    
    # Polarity Errors
    polarity_errors = tp[tp['polarity_gt'] != tp['polarity_pred']].copy()
    
    # Stats
    print(f"Total Ground Truth Aspects: {len(gt_df)}")
    print(f"Total Predicted Aspects: {len(pred_df)}")
    print(f"Correct Aspect Detection (TP): {len(tp)}")
    print(f"Missed Aspects (FN - False Negatives): {len(fn)}")
    print(f"Hallucinated Aspects (FP - False Positives): {len(fp)}")
    print(f"Polarity Mismatches (within correct aspects): {len(polarity_errors)}")
    
    # Save Examples
    with open(os.path.join(OUTPUT_REPORT_DIR, f'{model_name}_error_analysis.txt'), 'w', encoding='utf-8') as f:
        f.write(f"=== ERROR ANALYSIS FOR {model_name} ===\n\n")
        
        f.write(f"--- Top 10 False Positives (Hallucinated Aspects) ---\n")
        # Need text for context. FP rows have NaN text from GT if merge didn't carry it. 
        # But wait, we merged on ID/Aspect. If it's Right_only, we don't have the text row from GT unless we merge text separately.
        # Let's fix that by merging text back from GT based on ID.
        unique_texts = gt_df[['id', 'text']].drop_duplicates()
        fp = pd.merge(fp.drop(columns=['text']), unique_texts, on='id', how='left')
        
        for i, row in fp.head(10).iterrows():
            f.write(f"ID: {row['id']}\nText: {row['text']}\nPred Aspect: {row['aspectCategory']} (Not present in GT)\n\n")
            
        f.write(f"\n--- Top 10 False Negatives (Missed Aspects) ---\n")
        for i, row in fn.head(10).iterrows():
            f.write(f"ID: {row['id']}\nText: {row['text']}\nGT Aspect: {row['aspectCategory']} (Model missed this)\n\n")
            
        f.write(f"\n--- Top 10 Polarity Errors ---\n")
        for i, row in polarity_errors.head(15).iterrows():
            f.write(f"ID: {row['id']}\nText: {row['text']}\nAspect: {row['aspectCategory']}\nGT: {row['polarity_gt']} vs Pred: {row['polarity_pred']}\n\n")

    # Confusion Matrix for Polarity
    if len(tp) > 0 and PLOTS_AVAILABLE:
        labels = sorted(list(set(tp['polarity_gt'].unique()) | set(tp['polarity_pred'].unique())))
        cm = confusion_matrix(tp['polarity_gt'], tp['polarity_pred'], labels=labels)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
        plt.title(f'{model_name} Polarity Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Ground Truth')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_REPORT_DIR, f'{model_name}_confusion_matrix.png'))
        plt.close()
    elif not PLOTS_AVAILABLE:
        print(f"Skipping plots for {model_name} (dependencies missing).")

def main():
    try:
        gt, deberta, roberta = load_data()
        
        analyze_model("DeBERTa", deberta, gt)
        analyze_model("RoBERTa", roberta, gt)
        
        print(f"\nAnalysis complete! Check the '{OUTPUT_REPORT_DIR}' folder for text reports and plots.")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the paths to val_preds_*.csv are correct in the script.")

if __name__ == "__main__":
    main()
