#!/usr/bin/env python3
"""
NeuralNetwork class for binary classification with private attributes
"""

import numpy as np


class NeuralNetwork:
    """
    A neural network with one hidden layer performing binary classification
    """

    def __init__(self, nx, nodes):
        """
        Initialize a neural network

        Args:
            nx (int): Number of input features
            nodes (int): Number of nodes in the hidden layer

        Raises:
            TypeError: If nx is not an integer
            ValueError: If nx is less than 1
            TypeError: If nodes is not an integer
            ValueError: If nodes is less than 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Initialize private weights for hidden layer using random normal distribution
        self.__W1 = np.random.normal(0, 1, (nodes, nx))

        # Initialize private bias for hidden layer with 0's
        self.__b1 = np.zeros((nodes, 1))

        # Initialize private activated output for hidden layer
        self.__A1 = 0

        # Initialize private weights for output neuron using random normal distribution
        self.__W2 = np.random.normal(0, 1, (1, nodes))

        # Initialize private bias for output neuron
        self.__b2 = 0

        # Initialize private activated output for output neuron
        self.__A2 = 0

    @property
    def W1(self):
        """
        Getter for hidden layer weights

        Returns:
            numpy.ndarray: The hidden layer weights
        """
        return self.__W1

    @property
    def b1(self):
        """
        Getter for hidden layer bias

        Returns:
            numpy.ndarray: The hidden layer bias
        """
        return self.__b1

    @property
    def A1(self):
        """
        Getter for hidden layer activated output

        Returns:
            int: The hidden layer activated output
        """
        return self.__A1

    @property
    def W2(self):
        """
        Getter for output layer weights

        Returns:
            numpy.ndarray: The output layer weights
        """
        return self.__W2

    @property
    def b2(self):
        """
        Getter for output layer bias

        Returns:
            int: The output layer bias
        """
        return self.__b2

    @property
    def A2(self):
        """
        Getter for output layer activated output

        Returns:
            int: The output layer activated output
        """
        return self.__A2
