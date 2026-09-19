import pandas as pd
import sys

for path in sys.argv[1:]:
    print("=" * 60)
    print("FILE:", path)
    df = pd.read_csv(path)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())
    print("\nMissing values per column:")
    print(df.isna().sum())
    print()
