#!/usr/bin/env python3
"""
DeepNeuralNetwork class for binary classification
"""

import numpy as np


class DeepNeuralNetwork:
    """
    A deep neural network performing binary classification
    """

    def __init__(self, nx, layers):
        """
        Initialize a deep neural network

        Args:
            nx (int): Number of input features
            layers (list): List representing the number of nodes in each layer

        Raises:
            TypeError: If nx is not an integer
            ValueError: If nx is less than 1
            TypeError: If layers is not a list or empty
            TypeError: If elements in layers are not all positive integers
        """
        # Consolidated validation
        if not isinstance(nx, int) or nx < 1:
            raise TypeError("nx must be a positive integer") if not isinstance(nx, int) else ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(layer, int) and layer > 0 for layer in layers):
            raise TypeError("layers must be a list of positive integers")

        # Set the number of layers
        self.L = len(layers)

        # Initialize cache as empty dictionary
        self.cache = {}

        # Initialize weights dictionary
        self.weights = {}

        # Initialize weights and biases for each layer using vectorized approach
        for l in range(1, self.L + 1):
            # Previous layer size
            prev_layer = nx if l == 1 else layers[l - 2]
            current_layer = layers[l - 1]
            
            # Initialize weights using He et al. method
            # He initialization: std = sqrt(2/prev_layer)
            std = np.sqrt(2 / prev_layer)
            self.weights[f'W{l}'] = np.random.normal(0, std, (current_layer, prev_layer))
            
            # Initialize biases to 0's
            self.weights[f'b{l}'] = np.zeros((current_layer, 1))
