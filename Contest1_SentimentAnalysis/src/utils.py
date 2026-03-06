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
    outputs_dir = os.path.join(project_root, 'outputs')
    
    # Ensure outputs directory exists
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    
    paths = {
        'script_dir': script_dir,
        'project_root': project_root,
        'data_dir': data_dir,
        'outputs_dir': outputs_dir,
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
    output_path = os.path.join(paths['outputs_dir'], output_filename)
    
    # 1. Save Predictions
    df = pd.DataFrame(results_list)
    df.to_csv(output_path, index=False)
    print(f"\n[{split.upper()}] Predictions saved to {output_path}")
    
    # 2. Run Evaluation
    print(f"[{split.upper()}] Running evaluation...")
    csv_log_name = f"evaluation_{model_name.replace(' ', '_')}_log.csv"
    csv_log_path = os.path.join(paths['outputs_dir'], csv_log_name)
    
    subprocess.call([
        sys.executable,
        paths['eval_script'],
        paths['train_csv'],
        output_path,
        '--model', model_name,
        '--params', params,
        '--split', split,
        '--csv', csv_log_path
    ])

def save_submission(results_list, filename='submission.csv'):
    """
    Saves test set predictions for submission.
    """
    paths = get_paths()
    output_path = os.path.join(paths['outputs_dir'], filename)
    
    df = pd.DataFrame(results_list)
    df.to_csv(output_path, index=False)
    print(f"\n[TEST] Submission saved to {output_path}")
    
    # Verify format
    required_cols = {'id', 'aspectCategory', 'polarity'}
    if not required_cols.issubset(df.columns):
        print(f"WARNING: Submission missing columns. Found: {df.columns}")
    else:
        print("Submission format check: OK")
    
    print(df.head())
