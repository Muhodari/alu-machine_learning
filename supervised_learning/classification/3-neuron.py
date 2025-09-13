#!/usr/bin/env python3
"""
Neuron class for binary classification with cost calculation
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

    def forward_prop(self, X):
        """
        Calculate the forward propagation of the neuron

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
                nx is the number of input features
                m is the number of examples

        Returns:
            numpy.ndarray: The activated output (__A)
        """
        # Calculate the linear combination: Z = WX + b
        Z = np.dot(self.__W, X) + self.__b

        # Apply sigmoid activation function
        self.__A = 1 / (1 + np.exp(-Z))

        return self.__A

    def cost(self, Y, A):
        """
        Calculate the cost of the model using logistic regression

        Args:
            Y (numpy.ndarray): Correct labels with shape (1, m)
            A (numpy.ndarray): Activated output with shape (1, m)

        Returns:
            float: The cost
        """
        # Number of examples
        m = Y.shape[1]

        # Calculate the cost using logistic regression formula
        # Cost = -(1/m) * sum(Y * log(A) + (1-Y) * log(1-A))
        # Use 1.0000001 - A instead of 1 - A to avoid division by zero
        cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

        return cost
