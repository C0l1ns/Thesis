from sklearn.metrics import (
    r2_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score,
)
import numpy as np


def get_errors(y_target, y):
    def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(((y_pred - y_true) ** 2).mean())

    return [
        ("MAPE", str(mean_absolute_percentage_error(y_target, y))),
        ("RMSE", str(root_mean_squared_error(y_target, y))),
        ("MAE", str(mean_absolute_error(y_target, y))),
        ("Max error", str(max_error(y_target, y))),
        ("Median absolute error", str(median_absolute_error(y_target, y))),
        ("Mean Squared error", str(mean_squared_error(y_target, y))),
        ("R2", str(r2_score(y_target, y))),
    ]


def display_pretty_errors(errors: [str, str]):
    print()
    for name, value in errors:
        print(f"{name}: {value}")
    print()


def display_flat_errors(errors: [str, str]):
    print()
    for value in errors:
        print(value)
    print()
