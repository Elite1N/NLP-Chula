# Predictive Keyboard Contest - Experiment Instructions

This guide explains how to run the next-word prediction experiments, evaluate models, and generate submission files.

## 1. Quick Start

### Run LSTM Baseline (Recommended - Best Performance)

This trains a custom LSTM model on the provided training data.

**1. Train the model:**

```powershell
# Train on a subset (fast, for testing)
python src/train_lstm_full.py --epochs 1 --limit 500000

# Train on full dataset (slow, for final submission)
python src/train_lstm_full.py --epochs 5 --batch_size 2048
```

- **Outputs**: Saves model to `experiments/lstm_full/lstm_model_full.pth`

**2. Evaluate and Generate Submission:**

```powershell
python src/run_lstm_full.py
```

- **Outputs**:
  - `experiments/lstm_full/test_set_pred_full.txt` (Submission file)
  - `experiments/lstm_full/evaluations.csv` (Metrics)

### Run GPT-2 Baseline (Zero-Shot)

Uses specific pre-trained GPT-2 to predict the next word without training.

```powershell
python src/run_gpt2_baseline.py
```

- **Outputs**: `experiments/gpt2_baseline/evaluations.csv`

### Run RoBERTa Baseline (Experimental)

Fine-tunes a masked language model. _Note: Generally performs worse than LSTM for this specific generative task._

```powershell
# Train
python src/train_roberta.py --limit 50000 --epochs 1

# Evaluate
python src/run_roberta_baseline.py
```

## 2. Understanding the Logs

The scripts automatically log performance metrics to CSV files (e.g., `experiments/lstm_full/evaluations.csv`).

| Column               | Description                                           |
| :------------------- | :---------------------------------------------------- |
| **Model**            | Name of the model (e.g., "LSTM (Full)", "GPT-2").     |
| **Setting**          | Hyperparameters used (Embed size, Hidden dims, etc.). |
| **Training_Samples** | Number of lines used for training.                    |
| **Accuracy**         | The standard accuracy metric for the contest.         |

## 3. Submission

When you are ready to submit:

1.  Run the evaluation script for your best model (e.g., `src/run_lstm_full.py`).
2.  Locate the generated prediction file (e.g., `experiments/lstm_full/test_set_pred_full.txt`).
3.  Upload this `.txt` file to the contest platform.

## 4. Creating a New Model

To create a new model variation (e.g., `src/hybrid_model.py`), follow this pattern to ensure compatibility with the evaluation pipeline.

```python
import pandas as pd
from utils import evaluate_model, generate_test_predictions, log_experiment_result

class MyNewPredictor:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        # Load your model here
        pass

    def predict(self, context, first_letter):
        # 1. Logic to predict word
        # 2. Must start with first_letter
        return "predicted_word"

def main():
    # 1. Initialize Predictor
    predictor = MyNewPredictor("path/to/model")

    # 2. Evaluate on Dev Set
    dev_df = pd.read_csv('data/dev_set.csv')
    accuracy = evaluate_model(predictor, dev_df)
    print(f"Accuracy: {accuracy}")

    # 3. Log Results
    log_experiment_result('experiments/my_new_model', 'My Model', 'Params', 'Full Data', accuracy)

    # 4. Generate Submission
    test_df = pd.read_csv('data/test_set_no_answer.csv')
    output_path = 'experiments/my_new_model/submission.txt'
    generate_test_predictions(predictor, test_df, output_path)

if __name__ == "__main__":
    main()
```
