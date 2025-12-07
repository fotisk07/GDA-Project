import pandas as pd
import numpy as np
import torch
import falkon
from sklearn.metrics import roc_auc_score
import time

# ---- Streaming load ----
chunks = pd.read_csv("data/HIGGS.csv", header=None, chunksize=200_000)
X_all, y_all = [], []

for c in chunks:
    y_all.append(c.iloc[:,0].values)
    X_all.append(c.iloc[:,1:].values)

X = np.vstack(X_all).astype(np.float32)
y = np.hstack(y_all).astype(np.float32)

# ---- Split ----
idx = np.random.rand(len(X)) < 0.8
X_tr, X_ts = X[idx], X[~idx]
y_tr, y_ts = y[idx], y[~idx]

# ---- Torch ----
X_tr = torch.from_numpy(X_tr)
y_tr = torch.from_numpy(y_tr)
X_ts = torch.from_numpy(X_ts)
y_ts = torch.from_numpy(y_ts)

# ---- FALKON ----
opts = falkon.FalkonOptions(keops_active="no", use_cpu=False)

kernel = falkon.kernels.GaussianKernel(sigma=1.0, opt=opts)

model = falkon.Falkon(
    kernel=kernel,
    penalty=1e-4,
    M=5000,
    options=opts,
)

# ---- Train ----
t0 = time.time()
model.fit(X_tr, y_tr)
print("Train:", time.time()-t0)

# ---- Predict ----
t0 = time.time()
preds = model.predict(X_ts).cpu().numpy()
print("Infer:", time.time()-t0)

auc = roc_auc_score(y_ts.numpy(), preds)
print("AUC:", auc)
