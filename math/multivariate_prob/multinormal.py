#!/usr/bin/env python3
"""MultiNormal Distribution Class Module"""

import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal distribution
    """

    def __init__(self, data):
        """
        Initializes the distribution with data

        Parameters:
        - data: numpy.ndarray of shape (d, n)
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Mean vector of shape (d, 1)
        self.mean = np.mean(data, axis=1, keepdims=True)

        # Center the data
        X_centered = data - self.mean

        # Covariance matrix of shape (d, d)
        self.cov = (X_centered @ X_centered.T) / (n - 1)
