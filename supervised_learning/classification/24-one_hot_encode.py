#!/usr/bin/env python3
"""
One-hot encoding function
"""

import numpy as np


def one_hot_encode(Y, classes):
    """
    Convert a numeric label vector into a one-hot matrix

    Args:
        Y (numpy.ndarray): Numeric class labels with shape (m,)
        classes (int): Maximum number of classes found in Y

    Returns:
        numpy.ndarray: One-hot encoding of Y with shape (classes, m), or None on failure
    """
    if not isinstance(Y, np.ndarray) or Y.ndim != 1:
        return None
    
    if not isinstance(classes, int) or classes <= 0:
        return None
    
    # Check if all values in Y are valid (non-negative integers less than classes)
    if not np.all((Y >= 0) & (Y < classes) & (Y == Y.astype(int))):
        return None
    
    # Create one-hot encoding
    m = Y.shape[0]
    one_hot = np.zeros((classes, m))
    
    # Set the appropriate positions to 1
    one_hot[Y.astype(int), np.arange(m)] = 1
    
    return one_hot
