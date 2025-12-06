import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kernel_solvers import solve_falkon, solve_l2_problem, solve_l2_problem_Nystrom
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


def benchmark_vanilla_df(n_max=1e4, n_points=20):
    def vanilla_solver(X, y, m=None):
        solve_l2_problem(X, y)

    return run_benchmark(
        method="Vanilla",
        solver=vanilla_solver,
        n_max=n_max,
        n_points=n_points,
        filepath="bench_vanilla.csv",
    )


def benchmark_nystrom_df(n_max=1e4, n_points=20):
    def nystrom_solver(X, y, m):
        solve_l2_problem_Nystrom(X, y, m=m)

    return run_benchmark(
        method="Nystrom",
        solver=nystrom_solver,
        m_rule=lambda n: int(np.sqrt(n)),
        n_max=n_max,
        n_points=n_points,
        filepath="bench_nystrom.csv",
    )


def benchmark_falkon_df(n_max=1e4, n_points=20):
    def falkon_solver(X, y, m):
        solve_falkon(X, y, m=m)

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


# times, n_s = time_benchmark(n_max=1e4, n_points=20)
# n_s = np.asarray(n_s)
# times = np.asarray(times)

# fig, ax = plt.subplots(figsize=(8, 5))
# mask = n_s > 5e2
# n_ref = n_s[mask]
# t_ref = times[mask]

# # --- anchor on the FIRST asymptotic point ---
# n0 = n_ref[0]
# t0 = t_ref[0]

# on2 = t0 * (n_ref / n0) ** 2
# on3 = t0 * (n_ref / n0) ** 3

# # empirical data
# ax.loglog(
#     n_s,
#     times,
#     "o-",
#     linewidth=2,
#     markersize=6,
#     color="#1f77b4",
#     label="Measured runtime",
# )

# # reference slopes
# ax.loglog(n_ref, on2, "--", color="#FF7B9C", linewidth=2, label=r"$O(n^2)$")

# ax.loglog(n_ref, on3, "--", color="#FFC759", linewidth=2, label=r"$O(n^3)$")

# n_transition = 4e2
# ax.axvline(n_transition, color="grey", linestyle="--", linewidth=1.5, alpha=0.7)

# # Annotation
# ax.text(
#     0.05,
#     0.80,
#     "Python overhead regime",
#     transform=ax.transAxes,
#     fontsize=10,
#     color="grey",
#     ha="left",
#     va="top",
# )

# ax.set_xlabel("Dataset size $n$")
# ax.set_ylabel("Runtime (s)")
# ax.set_title("Kernel Method Scaling")

# ax.grid(True, which="both", linestyle=":", alpha=0.6)
# ax.legend(frameon=False)

# # Clean spines
# ax.spines["right"].set_visible(False)
# ax.spines["top"].set_visible(False)

# # Optional aesthetic tweak
# ax.spines["bottom"].set_bounds(n_s.min(), n_s.max())

# plt.tight_layout()
# plt.savefig("vanilla_scalling.png", dpi=300)
