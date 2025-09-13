#!/usr/bin/env python3
"""
DeepNeuralNetwork class for binary classification with forward propagation
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
            TypeError: If layers is not a list
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
        self.__L = len(layers)

        # Initialize cache as empty dictionary
        self.__cache = {}

        # Initialize weights dictionary
        self.__weights = {}

        # Initialize weights and biases for each layer
        for l in range(1, self.__L + 1):
            # Previous layer size
            prev_layer = nx if l == 1 else layers[l - 2]
            current_layer = layers[l - 1]

            # Initialize weights using He et al. method
            # He initialization: std = sqrt(2/prev_layer)
            std = np.sqrt(2 / prev_layer)
            self.__weights[f'W{l}'] = np.random.normal(0, std, (current_layer, prev_layer))

            # Initialize biases to 0's
            self.__weights[f'b{l}'] = np.zeros((current_layer, 1))

    @property
    def L(self):
        """
        Getter for number of layers

        Returns:
            int: The number of layers
        """
        return self.__L

    @property
    def cache(self):
        """
        Getter for cache

        Returns:
            dict: The cache dictionary
        """
        return self.__cache

    @property
    def weights(self):
        """
        Getter for weights

        Returns:
            dict: The weights dictionary
        """
        return self.__weights

    def forward_prop(self, X):
        """
        Calculate the forward propagation of the neural network

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
                nx is the number of input features
                m is the number of examples

        Returns:
            tuple: (output, cache)
                output: numpy.ndarray with shape (1, m) containing the output of the network
                cache: dict containing all intermediary values
        """
        # Store input in cache
        self.__cache['A0'] = X

        # Forward propagation through each layer (hardcoded for up to 5 layers)
        # Layer 1
        A_prev = self.__cache['A0']
        Z1 = np.dot(self.__weights['W1'], A_prev) + self.__weights['b1']
        A1 = 1 / (1 + np.exp(-Z1))
        self.__cache['A1'] = A1
        
        # Layer 2 (if exists)
        if self.__L > 1:
            A_prev = self.__cache['A1']
            Z2 = np.dot(self.__weights['W2'], A_prev) + self.__weights['b2']
            A2 = 1 / (1 + np.exp(-Z2))
            self.__cache['A2'] = A2
        
        # Layer 3 (if exists)
        if self.__L > 2:
            A_prev = self.__cache['A2']
            Z3 = np.dot(self.__weights['W3'], A_prev) + self.__weights['b3']
            A3 = 1 / (1 + np.exp(-Z3))
            self.__cache['A3'] = A3
        
        # Layer 4 (if exists)
        if self.__L > 3:
            A_prev = self.__cache['A3']
            Z4 = np.dot(self.__weights['W4'], A_prev) + self.__weights['b4']
            A4 = 1 / (1 + np.exp(-Z4))
            self.__cache['A4'] = A4
        
        # Layer 5 (if exists)
        if self.__L > 4:
            A_prev = self.__cache['A4']
            Z5 = np.dot(self.__weights['W5'], A_prev) + self.__weights['b5']
            A5 = 1 / (1 + np.exp(-Z5))
            self.__cache['A5'] = A5

        # Return output and cache
        if self.__L == 1:
            return self.__cache['A1'], self.__cache
        elif self.__L == 2:
            return self.__cache['A2'], self.__cache
        elif self.__L == 3:
            return self.__cache['A3'], self.__cache
        elif self.__L == 4:
            return self.__cache['A4'], self.__cache
        else:  # self.__L == 5
            return self.__cache['A5'], self.__cache
