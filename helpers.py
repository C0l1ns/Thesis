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


def insert_column(data: np.ndarray, column_to_insert, insert_idx):
    """
    Insert a new column into a NumPy array at a specified index.

    Parameters:
    data (np.ndarray): The original 2D NumPy array where the new column will be inserted.
    column_to_insert: The column to be inserted (1D NumPy array or list).
    insert_idx (int): The index at which the new column will be inserted.

    Returns:
    np.ndarray: A new NumPy array with the specified column inserted at the specified index.

    Notes:
    - The original array 'data' is not modified. Instead, a new array with the inserted column is returned.
    - 'insert_idx' should be a non-negative integer within the valid range of column indices for the input data.
    - 'column_to_insert' should have a compatible number of rows as 'data' for a successful insertion.
    - The 'axis=1' argument is used for inserting the column as a new column (horizontally).
    """
    new_array = np.insert(data, insert_idx, column_to_insert, axis=1)

    return new_array
