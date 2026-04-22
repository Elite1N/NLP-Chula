import pandas as pd

# Read CSV file into pandas DataFrame
test_df = pd.read_csv('data/raw/romanization-test-no-answer.csv')

# Read text file
with open('romanization-test-pred.txt', 'r', encoding='utf-8') as f:
    preds = f.readlines()

# Count rows
num_rows = len(test_df)
num_lines = len(preds)

if num_rows == num_lines:
    print('The number of rows in the DataFrame matches the number of lines in the text file.')
else:
    print('The number of rows in the DataFrame does not match the number of lines in the text file.')