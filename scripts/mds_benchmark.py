import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import falkon
from tqdm import tqdm
import time
from kernel_solvers import FalkonSolverGPU

def create_dataset(test_size:int = 0.1) -> tuple[torch.Tensor, torch.Tensor,torch.Tensor,torch.Tensor]:
    data_path = "data/YearPredictionMSD.txt"

    test_size = test_size
    random_state = 42

    ### Load the Data ###
    data = pd.read_csv(data_path,header=None)
    X,y = data.drop(0, axis=1), data[0].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=random_state,
)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    X_train = torch.tensor(X_train, dtype=torch.float64)
    y_train = torch.tensor(y_train, dtype=torch.float64)

    y_test = torch.tensor(y_test, dtype=torch.float64)
    scaler.fit(X_train)
    X_test = torch.tensor(scaler.transform(X_test))

    return X_train, y_train, X_test, y_test


def relative_error(y_pred, y_true):
    return torch.norm(y_pred - y_true) / torch.norm(y_true)

def run_benchmark(
    method:str,
    solver,
    start=2,
    stop=4,
    num_points=5,
    filepath=None,
):
    X_train, y_train, X_test, y_test = create_dataset()
    m_s = np.unique(np.logspace(2, stop, num_points).astype(int))
    records = []
    print(X_train.shape)

    print(f"Benchmarking {method}")
    print(m_s)

    for m in tqdm(m_s):
        t0 = time.time()
        solver(X_train, y_train,m)
        t = time.time() - t0

        y_pred = solver.predict(X_test)
        rel_err = relative_error(y_pred, y_test)
        
        records.append(
            {
                "method": method,
                "m": int(m),
                "time_sec": t,
                "rel_err" : rel_err.item()
            }
        )

    df = pd.DataFrame.from_records(records)

    if filepath is not None:
        df.to_csv(filepath, index=False)
        print(f"Saved {method} benchmark to: {filepath}")

    return df


def benchmark_falkonGPU(sigma, lam, start, stop, num_points):
    Falkon= FalkonSolverGPU(sigma, lam)
    return run_benchmark(
        method="FalkonGPU",
        solver=Falkon,
        start=start,
        stop=stop,
        num_points=num_points,
        filepath="outputs/MDS_FalkonGPU.csv",
    )



df_falkongpu = benchmark_falkonGPU(sigma=6, lam=1e-6, start=2, stop=4, num_points=5)