import time
import pandas as pd
import torch
from kmtr.datasets_and_metrics import BENCHMARKS
from kmtr.kernel_solvers import SOLVERS
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run kernel benchmark")

    parser.add_argument(
        "--dataset", type=str, default="MDS", help="Dataset name in BENCHMARKS dict"
    )

    parser.add_argument(
        "--model", type=str, default="FalkonGPU", help="Model name in SOLVERS dict"
    )

    parser.add_argument(
        "--m", type=int, default=int(1e4), help="Number of Nyström centers (m)"
    )

    parser.add_argument("--sigma", type=float, default=7.0, help="Kernel bandwidth")

    parser.add_argument(
        "--lam", type=float, default=2e-6, help="Regularization parameter"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path. If not provided, uses automatic naming.",
    )

    return parser.parse_args()


# -----------------
# Runner
# -----------------
def run(model, m, metric, Xtr, ytr, Xte, y_te):
    t0 = time.time()
    model.fit(Xtr, ytr, m)
    t_fit = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(Xte)
    t_pred = time.time() - t0

    err = metric(y_pred, y_te)
    if isinstance(err, torch.Tensor):
        err = err.item()

    print(f"train {t_fit:.2f}s | predict {t_pred:.2f}s | err {err:.5f}")

    return {
        "train_s": t_fit,
        "predict_s": t_pred,
        "total_s": t_fit + t_pred,
        "rel_err": err,
    }


def main():
    args = parse_args()

    dataset_name = args.dataset
    model_name = args.model
    m = args.m
    sigma = args.sigma
    lam = args.lam

    # Resolve dataset + metric
    dataset_fn = BENCHMARKS[dataset_name][0]
    metric_fn = BENCHMARKS[dataset_name][1]

    # Build model
    model = (
        SOLVERS[model_name](sigma, lam)
        if model_name != "GPytorch"
        else SOLVERS[model_name]()
    )

    # Output path
    output_path = (
        args.output
        if args.output is not None
        else f"outputs/{model_name}_{dataset_name}_{sigma}_{lam}_{m}.csv"
    )

    # ------------------------------------------------
    # Run benchmark
    # ------------------------------------------------
    print("Loading data...")
    X_tr, y_tr, X_te, y_te = dataset_fn()

    print("Starting run...")
    res = run(model, m, metric_fn, X_tr, y_tr, X_te, y_te)

    df = pd.DataFrame([res])
    df.to_csv(output_path, index=False)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
