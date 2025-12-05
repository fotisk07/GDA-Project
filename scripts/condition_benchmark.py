from kernel_solvers import rbf_kernel
import numpy as np
import matplotlib.pyplot as plt


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


n = 5
sigma = 1.0
reg = 1e-2
m = 50


X = np.linspace(-5, 5, n).reshape(-1, 1)
X_test = np.linspace(-6, 6, 100).reshape(-1, 1)

y_sin_star = np.sin(X).ravel()
y = y_sin_star + 0.2 * np.random.randn(n)
cond, n_s = condition_benchmark(stop=5, num=50)
# --- plot ---
fig, ax = plt.subplots(figsize=(7, 5))

ax.loglog(
    n_s,
    cond,
    linewidth=2.5,
    color="#1f77b4",  # deep blue
    marker="o",
    markersize=4,
    label="cond(A)",
)

ax.set_xlabel(r"Dataset size $n$")
ax.set_ylabel(r"Condition number of Nyström system")
ax.set_title("Conditioning of the Nyström Linear System")

ax.grid(True, which="major", alpha=0.4)
ax.grid(True, which="minor", alpha=0.2)

ax.legend(frameon=False)

plt.tight_layout()
plt.savefig("figures/conditioning.png", dpi=250)
