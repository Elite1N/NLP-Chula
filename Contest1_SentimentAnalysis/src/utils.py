import os
import sys
import pandas as pd
import subprocess

def get_paths():
    """
    Returns standard paths for the project relative to this script.
    """
    # Assuming utils.py is in src/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    
    paths = {
        'script_dir': script_dir,
        'project_root': project_root,
        'train_csv': os.path.join(data_dir, 'contest1_train.csv'),
        'test_csv': os.path.join(data_dir, 'contest1_test.csv'),
        'eval_script': os.path.join(script_dir, 'evaluate.py')
    }
    return paths

def save_and_evaluate(results_list, output_filename, model_name, params, split):
    """
    Saves predictions to CSV and immediately triggers the evaluation script.
    
    Args:
        results_list (list): List of dicts [{'id':..., 'aspectCategory':..., 'polarity':...}]
        output_filename (str): Name of the CSV file to save (e.g. 'val_preds.csv')
        model_name (str): Name for the log
        params (str): Parameters for the log
        split (str): 'train' or 'dev'
    """
    paths = get_paths()
    output_path = os.path.join(paths['project_root'], output_filename)
    
    # 1. Save Predictions
    df = pd.DataFrame(results_list)
    df.to_csv(output_path, index=False)
    print(f"\n[{split.upper()}] Predictions saved to {output_filename}")
    
    # 2. Run Evaluation
    print(f"[{split.upper()}] Running evaluation...")
    subprocess.call([
        sys.executable,
        paths['eval_script'],
        paths['train_csv'],
        output_path,
        '--model', model_name,
        '--params', params,
        '--split', split
    ])

def save_submission(results_list, filename='submission.csv'):
    """
    Saves test set predictions for submission.
    """
    paths = get_paths()
    output_path = os.path.join(paths['project_root'], filename)
    
    df = pd.DataFrame(results_list)
    df.to_csv(output_path, index=False)
    print(f"\n[TEST] Submission saved to {filename}")
    print(df.head())
