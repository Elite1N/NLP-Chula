# Sentiment Analysis Contest - Experiment Instructions

This guide explains how to run the sentiment analysis experiments and where to find the results.

## 1. Quick Start

### Run Logistic Regression Baseline (Fast)

```powershell
.venv/Scripts/python.exe src/baseline_logistic.py
```

- **Time**: ~10 seconds
- **Outputs**:
  - `val_preds_logistic.csv` (Validation predictions)
  - `train_preds_logistic.csv` (Training subset predictions)
  - `submission_baseline.csv` (Test set predictions for submission)
  - **Log**: `evaluation_Logistic_Baseline_log.csv`

### Run DistilBERT Baseline (Slow, Better)

```powershell
.venv/Scripts/python.exe src/baseline_bert.py
```

- **Time**: ~30-60 minutes (on CPU)
- **Outputs**:
  - `val_preds_bert.csv`
  - `train_preds_bert.csv`
  - `submission_bert.csv`
  - **Log**: `evaluation_DistilBERT_Baseline_log.csv`

## 2. Understanding the Logs

The scripts automatically log performance metrics to CSV files. Open `evaluation_Logistic_Baseline_log.csv` or `evaluation_DistilBERT_Baseline_log.csv` to see:

| Column           | Description                                                                 |
| :--------------- | :-------------------------------------------------------------------------- |
| **Split**        | `train` (performance on data it saw) vs `dev` (performance on unseen data). |
| **Aspect_F1**    | Score for correctly identifying aspects (Food, Service, etc.).              |
| **Sentiment_F1** | Score for correctly identifying sentiment (Pos, Neg, etc.).                 |
| **Overall_F1**   | The combined metric used for the contest. Focusing on this is key.          |

**Interpretation:**

- If `train` score is high but `dev` is low → **Overfitting**. (Need to simplify model or add regularization).
- If both `train` and `dev` are low → **Underfitting**. (Need a more complex model like BERT).

## 3. Manual Evaluation

If you want to manually evaluate a prediction file against the gold standard:

```powershell
# Evaluate on Dev Set
.venv/Scripts/python.exe src/evaluate.py data/contest1_train.csv val_preds_logistic.csv --model "MyCustomLogReg" --params "C=0.5" --split dev
```

## 4. Submission

When you are ready to submit:

1.  Run the training script.
2.  Take the generated `submission_*.csv` file.
3.  Upload it to the contest platform.

## 5. Creating a New Model

To create a new experiment while keeping the project organized, follow the standard template using `src/utils.py`.

1.  Create a new file, e.g., `src/experiment_lstm.py`.
2.  Use this template:

```python
import pandas as pd
import sys
import os

# Add src to path for utils import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import save_and_evaluate, save_submission, get_paths

def train_model():
    # 1. Setup
    paths = get_paths()
    TRAIN_FILE = paths['train_csv']
    TEST_FILE = paths['test_csv']

    MODEL_NAME = "My New Model"
    PARAMS = "Epochs=10, LR=0.001"

    # 2. Load & Prepare Data
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    # ... [YOUR TRAINING CODE HERE] ...

    # 3. Evaluate & Log (Validation)
    # val_results = [{'id': 101, 'aspectCategory': 'food', 'polarity': 'positive'}, ...]
    # save_and_evaluate(val_results, 'val_preds_new_model.csv', MODEL_NAME, PARAMS, 'dev')

    # 4. Generate Submission
    # test_results = ...
    # save_submission(test_results, 'submission_new_model.csv')

if __name__ == "__main__":
    train_model()
```

## 6. File Structure

- `src/baseline_logistic.py`: Light-weight baseline.
- `src/baseline_bert.py`: Deep Learning baseline.
- `src/evaluate.py`: The official scoring script (modified to support logging).
- `src/utils.py`: Helper functions for standardized I/O and evaluation.
- `data/`: Contains `contest1_train.csv` and `contest1_test.csv`.
