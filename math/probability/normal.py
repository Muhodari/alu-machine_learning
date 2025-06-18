#!/usr/bin/env python3
"""
Normal distribution module
"""


class Normal:
    """
    Represents a normal distribution
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initializes the normal distribution

        Args:
            data (list): list of the data to estimate the distribution
            mean (float): mean of the distribution (used if data is None)
            stddev (float): standard deviation (used if data is None)

        Raises:
            TypeError: if data is not a list
            ValueError: if stddev is not positive or data has < 2 values
        """
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
        """
        Calculates the z-score of a given x-value

        Args:
            x (float): the x-value

        Returns:
            float: the z-score corresponding to x
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score

        Args:
            z (float): the z-score

        Returns:
            float: the x-value corresponding to z
        """
        return self.mean + z * self.stddev

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value

        Args:
            x (float): x-value

        Returns:
            float: PDF value at x
        """
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
