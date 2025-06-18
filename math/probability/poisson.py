#!/usr/bin/env python3
"""
Poisson distribution module
"""


class Poisson:
    """
    Represents a Poisson distribution
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes the Poisson distribution

        Args:
            data (list): list of the data to estimate the distribution
            lambtha (float): expected number of occurrences in a time frame

        Raises:
            TypeError: if data is not a list
            ValueError: if lambtha is not positive or data has < 2 values
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of “successes”

        Args:
            k (int): number of “successes”

        Returns:
            PMF value for k
        """
        if k < 0:
            return 0

        k = int(k)

        e = 2.7182818285
        lambtha = self.lambtha

        # Compute k!
        fact = 1
        for i in range(1, k + 1):
            fact *= i

        # PMF formula: (λ^k * e^(-λ)) / k!
        pmf = (lambtha ** k) * (e ** -lambtha) / fact
        return pmf
