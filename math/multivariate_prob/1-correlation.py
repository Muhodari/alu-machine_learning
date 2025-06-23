#!/usr/bin/env python3
"""Correlation Matrix Calculation Module"""

import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix.

    Parameters:
    - C: numpy.ndarray of shape (d, d) - covariance matrix

    Returns:
    - numpy.ndarray of shape (d, d) containing the correlation matrix
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # Get standard deviations from the diagonal (variances)
    stddev = np.sqrt(np.diag(C))

    # Prevent division by zero
    if np.any(stddev == 0):
        raise ValueError("Standard deviation cannot be zero")

    # Compute outer product of stddev to form the denominator matrix
    denom = np.outer(stddev, stddev)

    # Element-wise division to get correlation matrix
    corr = C / denom

    return corr
