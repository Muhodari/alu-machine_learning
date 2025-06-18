#!/usr/bin/env python3
"""
Normal distribution module
"""


class Normal:
    """
    Represents a normal distribution
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = variance ** 0.5

    def z_score(self, x):
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        return self.mean + z * self.stddev

    def pdf(self, x):
        e = 2.7182818285
        pi = 3.1415926536
        part1 = 1 / (self.stddev * (2 * pi) ** 0.5)
        exponent = -0.5 * ((x - self.mean) / self.stddev) ** 2
        return part1 * (e ** exponent)

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value

        Args:
            x (float): x-value

        Returns:
            float: CDF value at x
        """
        pi = 3.1415926536
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))
        # Error function approximation
        erf = (2 / (pi ** 0.5)) * (z - (z ** 3) / 3 + (z ** 5) / 10 -
                                   (z ** 7) / 42 + (z ** 9) / 216)
        return 0.5 * (1 + erf)
