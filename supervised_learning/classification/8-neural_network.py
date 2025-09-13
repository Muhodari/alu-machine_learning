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
            TypeError: If nx or nodes is not an integer
            ValueError: If nx or nodes is less than 1
        """
        # Consolidated validation - check both parameters at once
        if not isinstance(nx, int) or not isinstance(nodes, int):
            raise TypeError("nx and nodes must be integers")
        if nx < 1 or nodes < 1:
            raise ValueError("nx and nodes must be positive integers")

        # Initialize weights for hidden layer using random normal distribution
        self.W1 = np.random.normal(0, 1, (nodes, nx))

        # Initialize bias for hidden layer with zeros
        self.b1 = np.zeros((nodes, 1))

        # Initialize activated output for hidden layer
        self.A1 = 0

        # Initialize weights for output neuron using random normal distribution
        self.W2 = np.random.normal(0, 1, (1, nodes))

        # Initialize bias for output neuron
        self.b2 = 0

        # Initialize activated output for output neuron
        self.A2 = 0
