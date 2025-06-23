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

        self.mean = np.mean(data, axis=1, keepdims=True)
        X_centered = data - self.mean
        self.cov = (X_centered @ X_centered.T) / (n - 1)
        self.d = d  # number of dimensions

    def pdf(self, x):
        """
        Calculates the PDF at a data point x

        Parameters:
        - x: numpy.ndarray of shape (d, 1)

        Returns:
        - The value of the PDF at x
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        if x.shape != (self.d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(self.d))

        diff = x - self.mean
        inv_cov = np.linalg.inv(self.cov)
        det_cov = np.linalg.det(self.cov)

        denom = np.sqrt(((2 * np.pi) ** self.d) * det_cov)
        exponent = -0.5 * (diff.T @ inv_cov @ diff)

        pdf_val = (1.0 / denom) * np.exp(exponent).item()
        return pdf_val
