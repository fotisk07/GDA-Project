import matplotlib.pyplot as plt
import numpy as np
from kernel_solvers import rbf_kernel


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


n_values = np.unique(np.logspace(2.7, 4.0, 8).astype(int))
m_values = {
    r"\sqrt{n}": np.sqrt(n_values).astype(int),
    r"n^{0.7}": (n_values**0.7).astype(int),
    r"n^{0.85}": (n_values**0.85).astype(int),
    r"0.1\,n": (0.1 * n_values).astype(int),
}


errors = {m: [] for m in m_values}

for label, m_vals in m_values.items():
    for n, m in zip(n_values, m_vals):
        err = nystrom_error(n, m)
        errors[label].append(err)

fig, ax = plt.subplots(figsize=(7, 5))

for m in m_values:
    ax.semilogx(
        n_values, errors[m], marker="o", linewidth=2, markersize=6, label=rf"$m={m}$"
    )

ax.set_xlabel(r"Dataset size $n$")
ax.set_ylabel(r"Relative error ")
ax.grid(True, which="major", alpha=0.4)
ax.grid(True, which="minor", alpha=0.2)
ax.set_title(r"Nyström approximation accuracy")

ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("figures/approximation.png", dpi=250)
