import pandas as pd

df = pd.read_csv("data/HIGGS.csv", header=None, dtype="float64")
df.columns = ["y"] + [f"x{i}" for i in range(1, df.shape[1])]

assert df.isna().sum().sum() == 0
assert all(dtype == "float64" for dtype in df.dtypes)

print("Loaded:", df.shape)

df.to_parquet("data/higgs.parquet", index=False)
print("Saved clean parquet")
