#!/usr/bin/env python3
"""
One-hot decoding function
"""

import numpy as np


def one_hot_decode(one_hot):
    """
    Convert a one-hot matrix into a vector of labels

    Args:
        one_hot (numpy.ndarray): One-hot encoded array with shape (classes, m)

    Returns:
        numpy.ndarray: Numeric labels with shape (m,), or None on failure
    """
    # Validate input type and dimensions
    if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
        return None
    
    # Combined validation: check if one_hot contains only 0s and 1s
    # and if each column has exactly one 1
    if not (np.all((one_hot == 0) | (one_hot == 1)) and 
            np.all(np.sum(one_hot, axis=0) == 1)):
        return None
    
    # Find the index of the maximum value in each column (argmax)
    # This is the most efficient way to decode one-hot encoding
    return np.argmax(one_hot, axis=0)
