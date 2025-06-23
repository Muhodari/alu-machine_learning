#!/usr/bin/env python3
"""
This module calculates the posterior probability distribution
for various hypothetical probabilities given observed data,
based on Bayes’ theorem.
"""

import numpy as np
import math


def factorial(n):
    """Compute factorial of a non-negative integer n."""
    return 1 if n == 0 else math.factorial(n)


def likelihood(x, n, P):
    """Helper function to calculate likelihoods."""
    comb_nx = factorial(n) / (factorial(x) * factorial(n - x))
    return comb_nx * (P ** x) * ((1 - P) ** (n - x))


def intersection(x, n, P, Pr):
    """
    Calculates the intersection of obtaining data x and n
    with each hypothetical probability in P and prior Pr.
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

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    return likelihood(x, n, P) * Pr


def marginal(x, n, P, Pr):
    """
    Calculates the marginal probability of obtaining x out of n
    using prior beliefs Pr and hypothetical probabilities P.
    """
    inter = intersection(x, n, P, Pr)
    return np.sum(inter)


def posterior(x, n, P, Pr):
    """
    Calculates the posterior probability for each hypothesis in P
    given observed data (x, n) and prior beliefs Pr.

    Returns:
    - np.ndarray of posterior probabilities.
    """
    inter = intersection(x, n, P, Pr)
    marg = marginal(x, n, P, Pr)
    return inter / marg
