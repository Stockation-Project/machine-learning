import pandas as pd
try:
    df = pd.read_parquet('clustering/idx80_raw.parquet')
    print("Successfully loaded parquet")
    print(df.head())
except Exception as e:
    print(f"Error: {e}")
