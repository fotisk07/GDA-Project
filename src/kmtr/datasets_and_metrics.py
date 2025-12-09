import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def MDS(
    test_size: int = 0.1, train_split: bool = True
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray]
):
    data_path = "data/YearPredictionMSD.txt"
    test_size = test_size
    random_state = 42

    ### Load the Data ###
    data = pd.read_csv(data_path, header=None)
    X, y = data.drop(0, axis=1), data[0].to_numpy()

    if not train_split:
        return X.to_numpy(), y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    scaler.fit(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


def Higgs(
    data_path="data/higgs.parquet", test_size: int = 0.2, train_split: bool = True
):
    data = pd.read_parquet(data_path)
    X, y = data.drop("y", axis=1), data["y"].to_numpy()
    if not train_split:
        return X.to_numpy(), y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    scaler.fit(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


def mini_higgs(test_size: int = 0.2, train_split: bool = True):
    return Higgs("data/higgs_mini.parquet", test_size, train_split)


def relative_error(y_pred, y_true):
    return np.abs((y_pred - y_true) / y_true).mean()


def one_minus_auc(y_pred, y_true):
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    auc = roc_auc_score(y_true, y_pred)
    return 1.0 - auc


BENCHMARKS = {
    "MSD": (MDS, relative_error),
    "HIGGS": (Higgs, one_minus_auc),
    "mini_Higgs": (mini_higgs, one_minus_auc),
}
