#!/usr/bin/env python3
"""
This module defines a function that calculates the definiteness of a matrix.
"""

import numpy as np

def definiteness(matrix):
    """
    Determines the definiteness of a square matrix.

    Args:
        matrix (np.ndarray): square matrix whose definiteness is to be determined

    Returns:
        str or None: one of 'Positive definite', 'Positive semi-definite',
                     'Negative definite', 'Negative semi-definite',
                     'Indefinite', or None
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.size == 0:
        return None

    try:
        eigvals = np.linalg.eigvalsh(matrix)  # more efficient for symmetric matrices
    except Exception:
        return None

    if np.all(eigvals > 0):
        return "Positive definite"
    elif np.all(eigvals >= 0):
        return "Positive semi-definite"
    elif np.all(eigvals < 0):
        return "Negative definite"
    elif np.all(eigvals <= 0):
        return "Negative semi-definite"
    elif np.any(eigvals > 0) and np.any(eigvals < 0):
        return "Indefinite"
    else:
        return None
