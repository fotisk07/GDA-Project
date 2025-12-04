import numpy as np


def rbf_kernel(X1, X2, sigma=0.6):
    # X1 (N, d)
    # X2 (N, d)
    sq_dist = (
        np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2 * (X1 @ X2.T)
    )
    return np.exp(-sq_dist / (2 * sigma**2))


def solve_l2_problem(X, y, sigma=0.1, reg=0.1):
    N, d = X.shape

    # Computer Kernel
    K = rbf_kernel(X, X, sigma=sigma)
    A = K + reg * N * np.eye(N)
    alpha_sol = np.linalg.solve(A, y)

    return alpha_sol


def solve_l2_problem_Nystrom(X, y, m=10, sigma=1.0, reg=1e-2):
    n = len(X)
    indices = np.random.choice(n, size=m, replace=False)
    Xm = X[indices]

    # Kernels
    K_mm = rbf_kernel(Xm, Xm, sigma)  # (m, m)
    K_mn = rbf_kernel(Xm, X, sigma)  # (m, n)

    # Nyström system:
    A = K_mn @ K_mn.T + reg * n * K_mm  # (m, m)
    b = K_mn @ y  # (m,)

    beta = np.linalg.solve(A, b)

    return beta, indices


def predict(X_test, X_train, coeff):
    K_test = rbf_kernel(X_test, X_train)
    return K_test @ coeff
