import numpy as np


def apply_min_max(data, min_val, max_val):
    """
    Applies min and max values to data.
    :param data:
    :param min_val:
    :param max_val:
    :return: scaled data
    """
    data = (data - min_val) / (max_val - min_val)
    return data


def reverse_min_max(scaled_data, min_val, max_val):
    """
    Reverses min-max scaling to recover the original data.

    Parameters:
    -----------
    scaled_data : array-like
        Data that has been scaled to the [0, 1] range.
    min_val : float or array-like
        The minimum value(s) used in the original scaling.
    max_val : float or array-like
        The maximum value(s) used in the original scaling.

    Returns:
    --------
    original_data : array-like
        The data mapped back to its original scale.
    """
    original_data = scaled_data * (max_val - min_val) + min_val
    return original_data