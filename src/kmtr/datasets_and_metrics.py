import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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


def relative_error(y_pred, y_true):
    return np.abs((y_pred - y_true) / y_true).mean()


BENCHMARKS = {"MDS": (MDS, relative_error)}
