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
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
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

        # Initialize weights and biases for each layer
        # Layer 1: input to first hidden layer
        std1 = np.sqrt(2 / nx)
        self.weights['W1'] = np.random.normal(0, std1, (layers[0], nx))
        self.weights['b1'] = np.zeros((layers[0], 1))
        
        # Additional layers (up to 4 more layers supported)
        if self.L > 1:
            std2 = np.sqrt(2 / layers[0])
            self.weights['W2'] = np.random.normal(0, std2, (layers[1], layers[0]))
            self.weights['b2'] = np.zeros((layers[1], 1))
        
        if self.L > 2:
            std3 = np.sqrt(2 / layers[1])
            self.weights['W3'] = np.random.normal(0, std3, (layers[2], layers[1]))
            self.weights['b3'] = np.zeros((layers[2], 1))
        
        if self.L > 3:
            std4 = np.sqrt(2 / layers[2])
            self.weights['W4'] = np.random.normal(0, std4, (layers[3], layers[2]))
            self.weights['b4'] = np.zeros((layers[3], 1))
        
        if self.L > 4:
            std5 = np.sqrt(2 / layers[3])
            self.weights['W5'] = np.random.normal(0, std5, (layers[4], layers[3]))
            self.weights['b5'] = np.zeros((layers[4], 1))
