import time

import numpy as np
import pandas as pd
from src.kernel_solvers import Falkon, Nystrom, VanillaKRR
from tqdm import tqdm


def run_benchmark(
    method,
    solver,
    n_max=1e4,
    n_points=20,
    m_rule=lambda n: None,  # for Nyström/Falkon
    filepath=None,
):
    n_s = np.unique(np.logspace(2, np.log10(n_max), n_points).astype(int))
    records = []

    print(f"Benchmarking {method} on sizes: {n_s}")

    for n in tqdm(n_s):
        X = np.linspace(-5, 5, n).reshape(-1, 1)
        y = np.sin(X).ravel() + 0.1 * np.random.randn(n)

        m = m_rule(n)

        t0 = time.time()
        solver(X, y, m)
        t = time.time() - t0

        records.append(
            {
                "method": method,
                "n": int(n),
                "m": None if m is None else int(m),
                "time_sec": t,
            }
        )

    df = pd.DataFrame.from_records(records)

    if filepath is not None:
        df.to_csv(filepath, index=False)
        print(f"Saved {method} benchmark to: {filepath}")

    return df


def benchmark_vanilla_df(n_max=1e4, n_points=20, sigma=1.0, lam=1e-2):
    def vanilla_solver(X, y, m=None):
        model = VanillaKRR(sigma=sigma, lam=lam)
        model.fit(X, y)

    return run_benchmark(
        method="Vanilla",
        solver=vanilla_solver,
        n_max=n_max,
        n_points=n_points,
        filepath="bench_vanilla.csv",
    )


def benchmark_nystrom_df(n_max=1e4, n_points=20, sigma=1.0, lam=1e-2):
    def nystrom_solver(X, y, m):
        model = Nystrom(sigma=sigma, lam=lam)
        model.fit(X, y, m)

    return run_benchmark(
        method="Nystrom",
        solver=nystrom_solver,
        m_rule=lambda n: int(np.sqrt(n)),
        n_max=n_max,
        n_points=n_points,
        filepath="bench_nystrom.csv",
    )


def benchmark_falkon_df(n_max=1e4, n_points=20, sigma=1.0, lam=1e-2):
    def falkon_solver(X, y, m):
        model = Falkon(sigma=sigma, lam=lam)
        model.fit(X, y, m)

    return run_benchmark(
        method="Falkon",
        solver=falkon_solver,
        m_rule=lambda n: int(np.sqrt(n)),
        n_max=n_max,
        n_points=n_points,
        filepath="bench_falkon.csv",
    )


df_vanilla = benchmark_vanilla_df(n_max=1e4, n_points=12)
df_nystrom = benchmark_nystrom_df(n_max=3e5, n_points=20)
df_falkon = benchmark_falkon_df(n_max=3e5, n_points=20)

df_all = pd.concat([df_vanilla, df_nystrom, df_falkon], ignore_index=True)
df_all.to_csv("bench_all_methods.csv", index=False)
