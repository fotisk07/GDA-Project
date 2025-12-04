import time
import numpy as np
import matplotlib.pyplot as plt
from kernel_solvers import solve_l2_problem


######### Benchmarking ############
def time_benchmark(n_max=1e3, n_points=2):
    # Log-spaced sizes from 10 to n_max
    n_s = np.unique(np.logspace(2, np.log10(n_max), n_points).astype(int))

    print(f"Benchmarking solve_l2_problem on dataset sizes: {n_s}")
    times = []
    n_s = [int(n) for n in n_s]
    X_benchmark = [np.linspace(-5, 5, n).reshape(-1, 1) for n in n_s]
    y_benchmark = [
        np.sin(X_benchmark[i]).ravel() + 0.1 * np.random.randn(n)
        for i, n in enumerate(n_s)
    ]

    times = []

    for i, n in enumerate(n_s):
        print(f"[{i}/{len(n_s)}] Running benchmark for n = {n}")
        t0 = time.time()
        solve_l2_problem(X_benchmark[i], y_benchmark[i])
        times.append((time.time() - t0))

    return times, n_s


times, n_s = time_benchmark(n_max=1e4, n_points=20)
n_s = np.asarray(n_s)
times = np.asarray(times)

fig, ax = plt.subplots(figsize=(8, 5))
mask = n_s > 5e2
n_ref = n_s[mask]
t_ref = times[mask]

# --- anchor on the FIRST asymptotic point ---
n0 = n_ref[0]
t0 = t_ref[0]

on2 = t0 * (n_ref / n0) ** 2
on3 = t0 * (n_ref / n0) ** 3

# empirical data
ax.loglog(
    n_s,
    times,
    "o-",
    linewidth=2,
    markersize=6,
    color="#1f77b4",
    label="Measured runtime",
)

# reference slopes
ax.loglog(n_ref, on2, "--", color="#FF7B9C", linewidth=2, label=r"$O(n^2)$")

ax.loglog(n_ref, on3, "--", color="#FFC759", linewidth=2, label=r"$O(n^3)$")

n_transition = 4e2
ax.axvline(n_transition, color="grey", linestyle="--", linewidth=1.5, alpha=0.7)

# Annotation
ax.text(
    0.05,
    0.80,
    "Python overhead regime",
    transform=ax.transAxes,
    fontsize=10,
    color="grey",
    ha="left",
    va="top",
)

ax.set_xlabel("Dataset size $n$")
ax.set_ylabel("Runtime (s)")  # <-- FIX #3
ax.set_title("Kernel Method Scaling")  # <-- FIX #2

ax.grid(True, which="both", linestyle=":", alpha=0.6)
ax.legend(frameon=False)

# Clean spines
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# Optional aesthetic tweak
ax.spines["bottom"].set_bounds(n_s.min(), n_s.max())  # <-- FIX #4

plt.tight_layout()
plt.savefig("vanilla_scalling.png", dpi=300)
