#!/usr/bin/env python3
"""
DeepNeuralNetwork class for binary classification with cost calculation
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

        # Forward propagation through each layer
        for l in range(1, self.__L + 1):
            # Get previous layer output
            A_prev = self.__cache[f'A{l-1}']

            # Calculate linear combination
            Z = np.dot(self.__weights[f'W{l}'], A_prev) + self.__weights[f'b{l}']

            # Apply sigmoid activation function
            A = 1 / (1 + np.exp(-Z))

            # Store in cache
            self.__cache[f'A{l}'] = A

        # Return output and cache
        return self.__cache[f'A{self.__L}'], self.__cache

    def cost(self, Y, A):
        """
        Calculate the cost of the model using logistic regression

        Args:
            Y (numpy.ndarray): Correct labels with shape (1, m)
            A (numpy.ndarray): Activated output with shape (1, m)

        Returns:
            float: The cost
        """
        # Number of examples
        m = Y.shape[1]

        # Calculate the cost using logistic regression formula
        # Cost = -(1/m) * sum(Y * log(A) + (1-Y) * log(1-A))
        # Use 1.0000001 - A instead of 1 - A to avoid division by zero
        cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

        return cost
