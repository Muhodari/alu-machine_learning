#!/usr/bin/env python3
"""
This module provides a function to calculate the likelihood of
observing a given number of patients with side effects from a
cancer drug trial, based on various hypothetical probabilities.
"""

import numpy as np
import math


def factorial(n):
    """Compute factorial of a non-negative integer n."""
    return 1 if n == 0 else math.factorial(n)


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining the data x and n
    for each probability in P.

    Parameters:
    - x (int): number of patients with side effects
    - n (int): total number of patients
    - P (np.ndarray): array of hypothetical probabilities

    Returns:
    - np.ndarray: likelihood values for each probability in P
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    comb_nx = factorial(n) / (factorial(x) * factorial(n - x))
    likelihoods = comb_nx * (P ** x) * ((1 - P) ** (n - x))

    return likelihoods
