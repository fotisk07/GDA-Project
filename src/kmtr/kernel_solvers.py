import gpytorch
import numpy as np
import torch
from scipy.linalg import solve_triangular
import falkon


def rbf_kernel(X1, X2, sigma=0.6):
    # X1 (N, d)
    # X2 (N, d)
    sq_dist = (
        np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2 * (X1 @ X2.T)
    )
    return np.exp(-sq_dist / (2 * sigma**2))


class VanillaKRR:
    def __init__(self, sigma: float = 1, lam: float = 0.1):
        self.sigma = sigma
        self.lam = lam

    def fit(self, X, y):
        N = len(X)
        K = rbf_kernel(X, X, sigma=self.sigma)
        A = K + self.lam * N * np.eye(N)
        self.coeff = np.linalg.solve(A, y)
        self.X_train = X

    def predict(self, X_test):
        K_test = rbf_kernel(X_test, self.X_train)
        return K_test @ self.coeff


class Nystrom:
    def __init__(self, sigma, lam):
        self.sigma = sigma
        self.lam = lam

    def fit(self, X, y, m):
        n = len(X)
        indices = np.random.choice(n, size=m, replace=False)
        Xm = X[indices]

        # Kernels
        K_mm = rbf_kernel(Xm, Xm, self.sigma)  # (m, m)
        K_mn = rbf_kernel(Xm, X, self.sigma)  # (m, n)

        # Nyström system:
        A = K_mn @ K_mn.T + self.lam * n * K_mm  # (m, m)
        b = K_mn @ y  # (m,)

        self.coeff = np.linalg.solve(A, b)
        self.Xm = Xm
        self.indices = indices

    def predict(self, X_test):
        K_test_m = rbf_kernel(X_test, self.Xm, self.sigma)
        return K_test_m @ self.coeff


class Falkon:
    def __init__(self, sigma, lam):
        self.sigma = sigma
        self.lam = lam

        self.coeff = None
        self.Xm = None

    @staticmethod
    def pcg(matvec, b, tol=1e-6, maxit=50):
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

    class _MatVec:
        def __init__(self, X, Xm, T, A, sigma, lam):
            self.X = X
            self.Xm = Xm
            self.T = T
            self.A = A
            self.sigma = sigma
            self.lam = lam

            self.n = len(X)

            # Precompute and cache kernels
            self.Knm = rbf_kernel(X, Xm, sigma=sigma)  # (n, m)
            self.Kmn = self.Knm.T  # (m, n)

        def apply(self, v):
            # v1 = A^{-1} v
            v1 = solve_triangular(self.A, v, lower=True)

            # v2 = T^{-1} v1
            v2 = solve_triangular(self.T, v1, lower=True)

            # kernel term
            tmp = self.Knm @ v2
            tmp = self.Kmn @ tmp

            # T^{-T}, A^{-T}
            tmp = solve_triangular(self.T.T, tmp, lower=False)
            tmp = solve_triangular(self.A.T, tmp, lower=False)

            out = tmp / self.n

            # regularization term  A^{-T}A^{-1}v
            reg = solve_triangular(self.A.T, v1, lower=False)
            out += self.lam * reg

            return out

    def fit(self, X, y, m):
        n = len(X)

        # 1) Landmark selection
        idx = np.random.choice(n, size=m, replace=False)
        Xm = X[idx]

        # 2) Kernel blocks
        Kmm = rbf_kernel(Xm, Xm, self.sigma)
        Kmn = rbf_kernel(Xm, X, self.sigma)

        # 3) Preconditioner Cholesky factors
        jitter = 1e-10 * np.trace(Kmm) / m
        T = np.linalg.cholesky(Kmm + jitter * np.eye(m))
        A = np.linalg.cholesky((T @ T.T) / m + self.lam * np.eye(m))

        # 4) RHS construction
        b = Kmn @ y

        b = solve_triangular(T.T, b, lower=False)
        b = solve_triangular(A.T, b, lower=False)
        b /= np.sqrt(n)

        # 5) PCG solve of preconditioned system
        mv = Falkon._MatVec(X, Xm, T, A, self.sigma, self.lam)
        beta = Falkon.pcg(mv.apply, b, maxit=40)

        # 6) Recover primal coefficients
        v = solve_triangular(A, beta, lower=True)
        v = solve_triangular(T, v, lower=True)

        alpha = v / np.sqrt(n)

        self.coeff = alpha
        self.Xm = Xm
        self.indices = idx

    def predict(self, X_test):
        if self.coeff is None:
            raise RuntimeError("Call fit() before predict().")

        K_test_m = rbf_kernel(X_test, self.Xm, self.sigma)
        return K_test_m @ self.coeff


class FalkonSolverGPU:
    def __init__(self, sigma, lam):
        self.options = falkon.FalkonOptions(keops_active="no")
        self.kernel = falkon.kernels.GaussianKernel(sigma=sigma, opt=self.options)
        self.lam = lam

    def fit(self, X, y, m):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        self.flk = falkon.Falkon(
            kernel=self.kernel, penalty=self.lam, M=m, options=self.options
        )
        self.flk.fit(X, y)

    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
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

    def fit(self, X, y, m):
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
        hyperparams = [
            p for name, p in self.model.named_parameters() if "variational" not in name
        ] + list(self.likelihood.parameters())

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


SOLVERS = {
    "Nystrom": Nystrom,
    "Falkon": Falkon,
    "FalkonGPU": FalkonSolverGPU,
}
