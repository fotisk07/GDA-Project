import numpy as np
from scipy.linalg import solve_triangular
import falkon
import torch
import gpytorch
import math

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

def predict_nystrom(X_test, Xm, beta, sigma):
    K_test_m = rbf_kernel(X_test, Xm, sigma)
    return K_test_m @ beta

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


class FalkonSolverGPU:
    def __init__(self, sigma, lam):
        self.options = falkon.FalkonOptions(keops_active="no")
        self.kernel = falkon.kernels.GaussianKernel(sigma=sigma, opt=self.options)
        self.lam = lam
    def __call__(self,X,y,m):
        self.flk = falkon.Falkon(kernel=self.kernel, penalty=self.lam, M=m, options=self.options)
        self.flk.fit(X,y)
    def predict(self,X):
        with torch.no_grad():
            return self.flk.predict(X).flatten()
        

class SVGPSolverGPyTorch:
    """
    GPyTorch SVGP solver wrapper with:
    - Gaussian likelihood (regression)
    - NGD for variational params
    - Adam for hyperparameters
    - Unwhitened inducing points
    """

    def __init__(
        self,
        num_iters=300,
        lr=0.05,
        use_cuda=True,
    ):
        self.num_iters = num_iters
        self.lr = lr
        self.device = (
            torch.device("cuda")
            if (use_cuda and torch.cuda.is_available())
            else torch.device("cpu")
        )

    # ---- SVGP Model definition ----
    class _SVGPModel(gpytorch.models.ApproximateGP):
        def __init__(self, inducing_points):
            variational_distribution = (
                gpytorch.variational.NaturalVariationalDistribution(
                    inducing_points.size(0)
                )
            )

            variational_strategy = gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            )

            super().__init__(variational_strategy)

            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # ---- Training call (same interface as FalkonSolverGPU) ----
    def __call__(self, X, y, m):
        X = X.to(self.device).float()
        y = y.to(self.device).float()

        n = X.shape[0]

        # Choose inducing points
        idx = torch.randperm(n, device=self.device)[:m]
        inducing_points = X[idx].clone()

        # Build model + likelihood
        self.model = self._SVGPModel(inducing_points).to(self.device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)

        self.model.train()
        self.likelihood.train()

        # ELBO
        mll = gpytorch.mlls.VariationalELBO(
            self.likelihood,
            self.model,
            num_data=n,
        )

        # --- split parameters for optimizers ---
        variational_params = list(self.model.variational_parameters())
        hyperparams = (
            [
                p for name, p in self.model.named_parameters()
                if "variational" not in name
            ]
            + list(self.likelihood.parameters())
        )

        nat_opt = gpytorch.optim.NGD(
            variational_params,
            lr=self.lr,
            num_data=n,
        )

        adam_opt = torch.optim.Adam(
            hyperparams,
            lr=self.lr,
        )

        # ---- training loop ----
        for _ in range(self.num_iters):
            nat_opt.zero_grad()
            adam_opt.zero_grad()

            out = self.model(X)
            loss = -mll(out, y)

            loss.backward()
            nat_opt.step()
            adam_opt.step()

        self.model.eval()
        self.likelihood.eval()

    # ---- Prediction ----
    @torch.no_grad()
    def predict(self, X):
        X = X.to(self.device).float()
        preds = self.likelihood(self.model(X))
        return preds.mean.cpu().flatten()
