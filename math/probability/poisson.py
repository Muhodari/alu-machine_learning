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
