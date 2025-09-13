#!/usr/bin/env python3
"""
DeepNeuralNetwork class for binary classification with private attributes
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
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list):
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
        # Layer 1: input to first hidden layer
        std1 = np.sqrt(2 / nx)
        self.__weights['W1'] = np.random.normal(0, std1, (layers[0], nx))
        self.__weights['b1'] = np.zeros((layers[0], 1))
        
        # Additional layers (up to 4 more layers supported)
        if self.__L > 1:
            std2 = np.sqrt(2 / layers[0])
            self.__weights['W2'] = np.random.normal(0, std2, (layers[1], layers[0]))
            self.__weights['b2'] = np.zeros((layers[1], 1))
        
        if self.__L > 2:
            std3 = np.sqrt(2 / layers[1])
            self.__weights['W3'] = np.random.normal(0, std3, (layers[2], layers[1]))
            self.__weights['b3'] = np.zeros((layers[2], 1))
        
        if self.__L > 3:
            std4 = np.sqrt(2 / layers[2])
            self.__weights['W4'] = np.random.normal(0, std4, (layers[3], layers[2]))
            self.__weights['b4'] = np.zeros((layers[3], 1))
        
        if self.__L > 4:
            std5 = np.sqrt(2 / layers[3])
            self.__weights['W5'] = np.random.normal(0, std5, (layers[4], layers[3]))
            self.__weights['b5'] = np.zeros((layers[4], 1))

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
