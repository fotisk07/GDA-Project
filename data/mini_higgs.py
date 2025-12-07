import sys
import pandas as pd

# usage: python make_mini_higgs.py 30000
N = int(sys.argv[1])
half = N // 2

df = pd.read_parquet("data/higgs.parquet")

label = df.columns[0]

mini = (
    pd.concat(
        [
            df[df[label] == 0].sample(half, random_state=13),
            df[df[label] == 1].sample(half, random_state=13),
        ]
    )
    .sample(frac=1, random_state=13)
    .reset_index(drop=True)
)

# ----------------------
# Set clean column names
# ----------------------
d = mini.shape[1] - 1  # number of features
mini.columns = ["y"] + [f"x{i + 1}" for i in range(d)]

mini.to_parquet("data/higgs_mini.parquet")

print(f"Saved higgs_mini_{N}.parquet ({len(mini)} rows)")
