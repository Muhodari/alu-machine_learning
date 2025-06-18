#!/usr/bin/env python3
"""
Exponential distribution module
"""


class Exponential:
    """
    Represents an exponential distribution
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes the Exponential distribution

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
            mean = sum(data) / len(data)
            self.lambtha = 1 / mean

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period

        Args:
            x (float): time period

        Returns:
            PDF value for x
        """
        if x < 0:
            return 0

        e = 2.7182818285
        return self.lambtha * (e ** (-self.lambtha * x))
