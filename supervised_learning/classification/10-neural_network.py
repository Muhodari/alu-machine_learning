#!/usr/bin/env python3
"""
NeuralNetwork class for binary classification with forward propagation
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
            TypeError: If nx or nodes is not an integer
            ValueError: If nx or nodes is less than 1
        """
        # Consolidated validation - check both parameters at once
        if not isinstance(nx, int) or not isinstance(nodes, int):
            raise TypeError("nx and nodes must be integers")
        if nx < 1 or nodes < 1:
            raise ValueError("nx and nodes must be positive integers")

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

    def forward_prop(self, X):
        """
        Calculate the forward propagation of the neural network

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
                nx is the number of input features
                m is the number of examples

        Returns:
            tuple: (A1, A2)
                A1: numpy.ndarray with shape (nodes, m) containing activated outputs of hidden layer
                A2: numpy.ndarray with shape (1, m) containing activated outputs of output layer
        """
        # Calculate the linear combination for hidden layer: Z1 = W1X + b1
        Z1 = np.dot(self.__W1, X) + self.__b1

        # Apply sigmoid activation function to hidden layer
        self.__A1 = 1 / (1 + np.exp(-Z1))

        # Calculate the linear combination for output layer: Z2 = W2A1 + b2
        Z2 = np.dot(self.__W2, self.__A1) + self.__b2

        # Apply sigmoid activation function to output layer
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2
