import numpy as np
from scipy.linalg import solve_triangular


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


class FalkonMatVec:
    def __init__(self, X, Xm, T, A, sigma, lam):
        self.X = X
        self.Xm = Xm
        self.T = T
        self.A = A
        self.lam = lam
        self.n = len(X)
        self.sigma = sigma

        self.Knm = rbf_kernel(X, Xm, sigma=sigma)  # (n,m)
        self.Kmn = self.Knm.T  # (m,n)

    def apply(self, v):
        """
        Computes:
        v -> P^T H P v   (no explicit matrices)
        """
        # v1 = A^{-1} v
        v1 = solve_triangular(self.A, v, lower=True)

        # v2 = T^{-1} v1
        v2 = solve_triangular(self.T, v1, lower=True)

        # kernel block:
        tmp = self.Knm @ v2  # (n,)
        tmp = self.Kmn @ tmp  # (m,)

        # T^{-T}, A^{-T}
        tmp = solve_triangular(self.T.T, tmp, lower=False)
        tmp = solve_triangular(self.A.T, tmp, lower=False)

        # main term
        out = tmp / self.n

        # regularization term: λ A^{-T}A^{-1} v
        reg = solve_triangular(self.A.T, v1, lower=False)
        out += self.lam * reg

        return out


def pcg(matvec, b, tol=1e-6, maxit=50):
    """
    Minimal preconditioned conjugate gradient
    """
    x = np.zeros_like(b)
    r = b - matvec(x)
    p = r.copy()
    rs = r @ r

    for k in range(maxit):
        Ap = matvec(p)

        alpha = rs / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap

        rs_new = r @ r
        if np.sqrt(rs_new) < tol:
            print(f"CG converged in {k + 1} iterations")
            break

        p = r + (rs_new / rs) * p
        rs = rs_new

    return x


def solve_falkon(X, y, m=100, sigma=1.0, lam=1e-2):
    n = len(X)

    # 1) landmark selection
    idx = np.random.choice(n, size=m, replace=False)
    Xm = X[idx]

    # 2) kernel blocks
    Kmm = rbf_kernel(Xm, Xm, sigma)
    Kmn = rbf_kernel(Xm, X, sigma)

    # 3) Cholesky factors
    jitter = 1e-10 * np.trace(Kmm) / m
    T = np.linalg.cholesky(Kmm + jitter * np.eye(m))

    A = np.linalg.cholesky((T @ T.T) / m + lam * np.eye(m))

    # 4) RHS: b = P^T Kmn y
    # here: b_tilde = A^{-T} T^{-T} Kmn y / sqrt(n)
    b = Kmn @ y

    b = solve_triangular(T.T, b, lower=False)
    b = solve_triangular(A.T, b, lower=False)
    b = b / np.sqrt(n)

    # 5) Matrix-free CG operator
    mv = FalkonMatVec(X, Xm, T, A, sigma, lam)
    beta = pcg(mv.apply, b, maxit=40)

    # 6) recover alpha = P beta
    v = solve_triangular(A, beta, lower=True)
    v = solve_triangular(T, v, lower=True)

    alpha = v / np.sqrt(n)

    return alpha, idx
