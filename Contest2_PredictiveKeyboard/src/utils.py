import time
import pandas as pd

def load_training_data(filepath, limit=None):
    print(f"Loading training data from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = []
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            lines.append(line.strip())
    print(f"Loaded {len(lines)} lines.")
    return lines

def evaluate_model(model, df):
    correct = 0
    total = len(df)
    
    print(f"Evaluating on {total} examples...")
    start_eval = time.time()
    for index, row in df.iterrows():
        context = row['context']
        first_letter = row['first letter']
        true_answer = row['answer']
        
        prediction = model.predict(context, first_letter)
        
        if prediction == true_answer:
            correct += 1
            
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Evaluation time: {time.time() - start_eval:.2f}s")
    return accuracy

def generate_test_predictions(model, test_df, output_path):
    print(f"Generating predictions for {len(test_df)} test examples...")
    predictions = []
    start_time = time.time()
    
    for index, row in test_df.iterrows():
        context = row['context']
        first_letter = row['first letter']
        
        prediction = model.predict(context, first_letter)
        predictions.append(prediction)
        
    print(f"Prediction finished in {time.time() - start_time:.2f} s")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    print(f"Saved predictions to {output_path}")
