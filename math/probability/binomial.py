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

    def pmf(self, k):
        """
        Calculates the PMF for a given number of successes

        Args:
            k (int): number of successes

        Returns:
            float: PMF value for k
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        return (self._factorial(self.n) /
                (self._factorial(k) * self._factorial(self.n - k))) * \
               (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def _factorial(self, num):
        """
        Calculates factorial of a number

        Args:
            num (int): the number

        Returns:
            int: factorial of num
        """
        if num == 0 or num == 1:
            return 1
        result = 1
        for i in range(2, num + 1):
            result *= i
        return result
