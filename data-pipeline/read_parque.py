import pandas as pd

df = pd.read_parquet("data/03_primary/filing_data.parquet")
print(df.head())
print(df.columns)
