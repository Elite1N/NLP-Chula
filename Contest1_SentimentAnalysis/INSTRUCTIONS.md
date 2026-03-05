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

## 5. File Structure

- `src/baseline_logistic.py`: Light-weight baseline.
- `src/baseline_bert.py`: Deep Learning baseline.
- `src/evaluate.py`: The official scoring script (modified to support logging).
- `data/`: Contains `contest1_train.csv` and `contest1_test.csv`.
