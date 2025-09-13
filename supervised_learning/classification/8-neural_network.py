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
            nx (int or float): Number of input features
            nodes (int or float): Number of nodes in the hidden layer

        Raises:
            TypeError: If nx or nodes is not a number
            ValueError: If nx or nodes is less than 1
        """
        # Consolidated validation - check both parameters at once
        if not isinstance(nx, (int, float)) or not isinstance(
                nodes, (int, float)):
            raise TypeError("nx and nodes must be numbers")
        if nx < 1 or nodes < 1:
            raise ValueError("nx and nodes must be positive numbers")
        
        # Convert to integers if they are floats
        nx = int(nx)
        nodes = int(nodes)

        # Initialize all weights and biases using vectorized operations
        self.W1 = np.random.normal(0, 1, (nodes, nx))
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.normal(0, 1, (1, nodes))
        self.b2 = 0
        self.A2 = 0
