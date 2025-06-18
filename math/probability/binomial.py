#!/usr/bin/env python3
"""
Binomial distribution module
"""


class Binomial:
    """
    Represents a binomial distribution
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes the binomial distribution

        Args:
            data (list): list of observed data
            n (int): number of trials (if no data)
            p (float): probability of success (if no data)

        Raises:
            TypeError: if data is not a list
            ValueError: if data has < 2 values,
                        if n is not positive,
                        if p is not in (0, 1)
        """
        if data is None:
            if not isinstance(n, int) or n < 1:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = n
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            var = sum((x - mean) ** 2 for x in data) / len(data)
            p_est = 1 - (var / mean)
            n_est = round(mean / p_est)
            self.n = n_est
            self.p = mean / self.n
