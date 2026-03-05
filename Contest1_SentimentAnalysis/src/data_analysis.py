import pandas as pd

try:
    df = pd.read_csv('./data/contest1_train.csv')
    print("Columns:", df.columns)
    print("\nUnique Aspects:")
    print(df['aspectCategory'].unique())
    print("\nAspect Counts:")
    print(df['aspectCategory'].value_counts())
    
    print("\nUnique Polarities:")
    print(df['polarity'].unique())
    print("\nPolarity Counts:")
    print(df['polarity'].value_counts())
    
    print("\nSample Rows:")
    print(df.head())
    
    print("\nTotal Rows:", len(df))
    print("Total Unique IDs:", df['id'].nunique())
except Exception as e:
    print("Error:", e)
