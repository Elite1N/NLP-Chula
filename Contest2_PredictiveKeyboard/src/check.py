import pandas as pd
TEST_PATH = '../data/test_set_no_answer.csv'
SUBMISSION_PATH = '../data/test_set_pred.txt'

# read CSV file into pandas DataFrame
df = pd.read_csv(TEST_PATH)

# read text file with one word per line
with open(SUBMISSION_PATH) as f:
    words = f.readlines()
    num_lines = len(words)


# compare number of rows in DataFrame with number of lines in text file
num_rows = len(df)
if num_rows == num_lines:
    print('The number of rows in the DataFrame matches the number of lines in the text file.')
else:
    print('The number of rows in the DataFrame does not match the number of lines in the text file.')
