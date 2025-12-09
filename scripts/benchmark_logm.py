import os
import time
import pandas as pd
import torch
import argparse
import numpy as np

from kmtr.datasets_and_metrics import BENCHMARKS
from kmtr.kernel_solvers import SOLVERS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run kernel benchmark over log-range of m"
    )

    parser.add_argument("--dataset", type=str, default="MDS")
    parser.add_argument("--model", type=str, default="FalkonGPU")
    parser.add_argument(
        "--m_stop",
        type=int,
        required=True,
        help="Max Nyström centers (log sweep stops here)",
    )
    parser.add_argument("--sigma", type=float, default=7.0)
    parser.add_argument("--lam", type=float, default=2e-6)
    parser.add_argument("--num", type=int, default=10)

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default auto-named)",
    )

    return parser.parse_args()


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

    print(f"m={m:6d} | train {t_fit:.2f}s | predict {t_pred:.2f}s | err {err:.5f}")

    return {
        "m": m,
        "train_s": t_fit,
        "predict_s": t_pred,
        "total_s": t_fit + t_pred,
        "rel_err": err,
    }


def main():
    args = parse_args()

    dataset_name = args.dataset
    model_name = args.model
    sigma = args.sigma
    lam = args.lam
    n_points = args.num

    # ------------------------------------------------
    # Log-scale sweep of m
    # ------------------------------------------------
    m_start = 100
    m_stop = args.m_stop

    ms = np.unique(
        np.logspace(
            np.log10(m_start),
            np.log10(m_stop),
            num=n_points,
            dtype=int,
        )
    )

    # ------------------------------------------------
    # Dataset + metric
    # ------------------------------------------------
    dataset_fn, metric_fn = BENCHMARKS[dataset_name]

    # ------------------------------------------------
    # Output file
    # ------------------------------------------------
    output_path = (
        args.output
        if args.output is not None
        else f"outputs/{dataset_name}/{model_name}_{sigma}_{lam}_logMs.csv"
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Always reset the file at start
    pd.DataFrame(columns=["m", "train_s", "predict_s", "total_s", "rel_err"]).to_csv(
        output_path, index=False
    )

    # Write CSV header once if file doesn't exist
    if not os.path.exists(output_path):
        pd.DataFrame(
            columns=["m", "train_s", "predict_s", "total_s", "rel_err"]
        ).to_csv(output_path, index=False)

    # ------------------------------------------------
    # Load data once
    # ------------------------------------------------
    print("Loading data...")
    X_tr, y_tr, X_te, y_te = dataset_fn()

    # ------------------------------------------------
    # Sweep
    # ------------------------------------------------
    for m in ms:
        print(f"\n=== Running m = {m} ===")

        try:
            model = (
                SOLVERS[model_name](sigma, lam)
                if model_name != "GPytorch"
                else SOLVERS[model_name]()
            )
            res = run(model, m, metric_fn, X_tr, y_tr, X_te, y_te)

            # Append result immediately
            pd.DataFrame([res]).to_csv(
                output_path,
                mode="a",
                header=False,
                index=False,
            )

            print(f"Saved result for m={m}")

        except Exception as e:
            print(f"FAILED at m={m} — error: {e}")
            continue


if __name__ == "__main__":
    main()
