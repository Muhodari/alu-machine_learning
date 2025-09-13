#!/usr/bin/env python3
"""
Neuron class for binary classification with private attributes
"""

import numpy as np


class Neuron:
    """
    A single neuron performing binary classification
    """

    def __init__(self, nx):
        """
        Initialize a neuron

        Args:
            nx (int): Number of input features to the neuron

        Raises:
            TypeError: If nx is not an integer
            ValueError: If nx is less than 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        # Initialize private weights using random normal distribution
        self.__W = np.random.normal(0, 1, (1, nx))

        # Initialize private bias to 0
        self.__b = 0

        # Initialize private activated output to 0
        self.__A = 0

    @property
    def W(self):
        """
        Getter for weights vector

        Returns:
            numpy.ndarray: The weights vector
        """
        return self.__W

    @property
    def b(self):
        """
        Getter for bias

        Returns:
            int: The bias value
        """
        return self.__b

    @property
    def A(self):
        """
        Getter for activated output

        Returns:
            int: The activated output value
        """
        return self.__A
