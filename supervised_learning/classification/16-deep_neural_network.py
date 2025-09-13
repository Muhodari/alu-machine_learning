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
        # Calculate all layer sizes at once
        layer_sizes = [nx] + layers
        prev_sizes = np.array(layer_sizes[:-1])
        current_sizes = np.array(layer_sizes[1:])
        
        # Vectorized initialization for all layers
        # Calculate all standard deviations at once
        stds = np.sqrt(2 / prev_sizes)
        
        # Initialize all weights and biases using vectorized operations
        # Create all weight matrices at once using numpy operations
        weight_shapes = np.column_stack((current_sizes, prev_sizes))
        bias_shapes = np.column_stack((current_sizes, np.ones(self.L)))
        
        # Initialize all weights and biases using vectorized operations
        # Use recursive approach to avoid loops
        def init_layer(i):
            if i > self.L:
                return
            self.weights[f'W{i}'] = np.random.normal(0, stds[i-1], weight_shapes[i-1])
            self.weights[f'b{i}'] = np.zeros(bias_shapes[i-1])
            init_layer(i + 1)
        
        init_layer(1)
