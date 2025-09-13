#!/usr/bin/env python3
"""
NeuralNetwork class for binary classification with one hidden layer
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

        # Initialize weights for hidden layer using random normal distribution
        self.W1 = np.random.normal(0, 1, (nodes, nx))

        # Initialize bias for hidden layer with 0's
        self.b1 = np.zeros((nodes, 1))

        # Initialize activated output for hidden layer
        self.A1 = 0

        # Initialize weights for output neuron using random normal distribution
        self.W2 = np.random.normal(0, 1, (1, nodes))

        # Initialize bias for output neuron
        self.b2 = 0

        # Initialize activated output for output neuron
        self.A2 = 0
