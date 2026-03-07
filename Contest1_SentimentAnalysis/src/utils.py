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

# ---------------------------------------------------------
# Heuristic Keywords for Aspect Detection
# ---------------------------------------------------------
# These keywords are used to override or supplement the model's predictions.
# If the model misses a class but a strong keyword is present, we add the class.
HEURISTIC_KEYWORDS = {
    'price': ['price', 'cost', 'expensive', 'cheap', 'bill', 'check', 'worth', 'value', 'overpriced', 'reasonable', '$', 'money', 'paid', 'pay', 'thb', 'baht'],
    'service': ['service', 'waiter', 'waitress', 'staff', 'manager', 'server', 'served', 'rude', 'friendly', 'attitude', 'ignored', 'slow', 'fast', 'attentive', 'greeting'],
    'ambience': ['ambience', 'atmosphere', 'decor', 'view', 'music', 'noisy', 'loud', 'quiet', 'crowded', 'clean', 'dirty', 'comfortable', 'seating', 'vibe', 'interior', 'design', 'decoration'],
    'food': ['food', 'delicious', 'tasty', 'yummy', 'bland', 'flavor', 'meat', 'chicken', 'fish', 'pork', 'beef', 'salad', 'soup', 'dessert', 'drink', 'beverage', 'wine', 'beer', 'menu', 'dish', 'portion', 'fresh'],
}

def apply_heuristics(text, predicted_aspects):
    """
    Applies keyword-based heuristics to ensure critical aspects are not missed.
    This acts as a high-precision, low-recall safety net (or high-recall, low-precision depending on keyword quality).
    """
    text_lower = text.lower()
    
    # Check Price
    if 'price' not in predicted_aspects:
        if any(k in text_lower for k in HEURISTIC_KEYWORDS['price']):
             predicted_aspects.append('price')
             
    # Check Service
    if 'service' not in predicted_aspects:
         if any(k in text_lower for k in HEURISTIC_KEYWORDS['service']):
             predicted_aspects.append('service')
             
    # Check Ambience
    if 'ambience' not in predicted_aspects:
         if any(k in text_lower for k in HEURISTIC_KEYWORDS['ambience']):
             predicted_aspects.append('ambience')
             
    # Check Food (Default fallback if nothing else is found, or if strong food keywords exist)
    if not predicted_aspects and any(k in text_lower for k in HEURISTIC_KEYWORDS['food']):
        predicted_aspects.append('food')

    return list(set(predicted_aspects))

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
