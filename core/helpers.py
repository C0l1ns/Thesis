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


def collect_cluster_center_target_coordinates(
    cluster_centers: np.ndarray, best_labels: np.ndarray, feature_offset: int = -1
):
    """
    Extracts and returns the target coordinates (y values) of
    cluster centers for given labels.
    """
    extracted_coordinate = cluster_centers[:, -1:]

    cluster_center_feature = [
        extracted_coordinate[cluster_idx] for cluster_idx in best_labels
    ]

    return np.array(cluster_center_feature)


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(((y_pred - y_true) ** 2).mean())


def flat_errors(y_target, y):
    return [
        ("MAPE", str(mean_absolute_percentage_error(y_target, y))),
        ("RMSE", str(root_mean_squared_error(y_target, y))),
        ("MAE", str(mean_absolute_error(y_target, y))),
        ("Max error", str(max_error(y_target, y))),
        ("Median absolute error", str(median_absolute_error(y_target, y))),
        ("Mean Squared error", str(mean_squared_error(y_target, y))),
        ("R2", str(r2_score(y_target, y))),
    ]


def find_yn(z, y_sum, N):
    return np.array([(y_sum + sum(z[i : i + N])) / N for i in range(0, len(z), N)])
