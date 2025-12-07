import numpy as np
from kernel_solvers import rbf_kernel
import pandas as pd


def nystrom_error(n, m, sigma=1.0, seed=0):
    rng = np.random.default_rng(seed)

    X = np.linspace(-5, 5, n).reshape(-1, 1)
    idx = rng.choice(n, size=m, replace=False)
    Xm = X[idx]

    K_nm = rbf_kernel(X, Xm, sigma)
    K_mm = rbf_kernel(Xm, Xm, sigma)

    A_full = K_nm.T @ K_nm
    A_approx = (n / m) * (K_mm @ K_mm)

    return np.linalg.norm(A_full - A_approx, ord="fro") / np.linalg.norm(
        A_full, ord="fro"
    )


def condition_benchmark(start=1, stop=3, num=100):
    n_s = np.logspace(start, stop, num)
    cond = []
    for n in n_s:
        m = int(np.sqrt(n))
        X = np.linspace(-5, 5, int(n)).reshape(-1, 1)
        indices = np.random.choice(int(n), size=m, replace=False)
        Xm = X[indices]
        # Kernels
        K_mm = rbf_kernel(Xm, Xm, sigma)  # (m, m)
        K_mn = rbf_kernel(Xm, X, sigma)  # (m, n)

        # Nyström system:
        A = K_mn @ K_mn.T + reg * n * K_mm  # (m, m)

        cond.append(np.linalg.cond(A))

    return cond, n_s


############ Nystrom Error ################
n_values = np.unique(np.logspace(2.7, 4.0, 8).astype(int))
m_values = {
    r"\sqrt{n}": np.sqrt(n_values).astype(int),
    r"n^{0.7}": (n_values**0.7).astype(int),
    r"n^{0.85}": (n_values**0.85).astype(int),
    r"0.1\,n": (0.1 * n_values).astype(int),
}


df = pd.DataFrame(
    [
        {
            "scaling": label,
            "n": int(n),
            "m": int(m),
            "rel_fro_error": nystrom_error(n, m),
        }
        for label, m_vals in m_values.items()
        for n, m in zip(n_values, m_vals)
    ]
)

df.to_csv("outputs/nystrom_error_scaling.csv", index=False)


############### Condition Benchamrks #################
n = 5
sigma = 1.0
reg = 1e-2
m = 50

X = np.linspace(-5, 5, n).reshape(-1, 1)
X_test = np.linspace(-6, 6, 100).reshape(-1, 1)

y_sin_star = np.sin(X).ravel()
y = y_sin_star + 0.2 * np.random.randn(n)
cond, n_s = condition_benchmark(stop=5, num=50)
pd.DataFrame({"n": n_s, "condition_number": cond}).to_csv(
    "outputs/condition_benchmark.csv", index=False
)
