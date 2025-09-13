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

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
nn = NeuralNetwork(X.shape[0], 3)
print(nn.W1)
print(nn.W1.shape)
print(nn.b1)
print(nn.W2)
print(nn.W2.shape)
print(nn.b2)
print(nn.A1)
print(nn.A2)
nn.A1 = 10
print(nn.A1)
