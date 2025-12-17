import pandas as pd

def get_schema(csv_path="data.csv"):
    df = pd.read_csv(csv_path, encoding="latin1", nrows=5)
    return list(df.columns)
