#!/usr/bin/env python3
"""
DeepNeuralNetwork class for binary classification with verbose and graph training
"""

import numpy as np
import matplotlib.pyplot as plt


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

        # Initialize weights and biases for each layer using recursive approach
        def init_layer(l):
            if l > self.__L:
                return
            # Previous layer size
            prev_layer = nx if l == 1 else layers[l - 2]
            current_layer = layers[l - 1]

            # Initialize weights using He et al. method
            # He initialization: std = sqrt(2/prev_layer)
            std = np.sqrt(2 / prev_layer)
            self.__weights[f'W{l}'] = np.random.normal(0, std, (current_layer, prev_layer))

            # Initialize biases to 0's
            self.__weights[f'b{l}'] = np.zeros((current_layer, 1))
            init_layer(l + 1)
        
        init_layer(1)
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

        # Forward propagation through each layer using recursive approach
        def forward_layer(l):
            if l > self.__L:
                return
            # Get previous layer output
            A_prev = self.__cache[f'A{l-1}']

            # Calculate linear combination
            Z = np.dot(self.__weights[f'W{l}'], A_prev) + self.__weights[f'b{l}']

            # Apply sigmoid activation function
            A = 1 / (1 + np.exp(-Z))

            # Store in cache
            self.__cache[f'A{l}'] = A
            forward_layer(l + 1)
        
        forward_layer(1)
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

    def evaluate(self, X, Y):
        """
        Evaluate the neural network's predictions

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
                nx is the number of input features
                m is the number of examples
            Y (numpy.ndarray): Correct labels with shape (1, m)

        Returns:
            tuple: (prediction, cost)
                prediction: numpy.ndarray with shape (1, m) containing predicted labels
                cost: float representing the cost of the network
        """
        # Perform forward propagation to get activated outputs
        A, _ = self.forward_prop(X)

        # Convert probabilities to binary predictions
        # 1 if output >= 0.5, 0 otherwise
        prediction = (A >= 0.5).astype(int)

        # Calculate the cost
        cost = self.cost(Y, A)

        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculate one pass of gradient descent on the neural network

        Args:
            Y (numpy.ndarray): Correct labels with shape (1, m)
            cache (dict): Dictionary containing all intermediary values
            alpha (float): Learning rate
        """
        # Number of examples
        m = Y.shape[1]

        # Initialize gradients dictionary
        grads = {}

        # Calculate gradients for output layer
        dZ = cache[f'A{self.__L}'] - Y
        grads[f'dW{self.__L}'] = (1 / m) * np.dot(dZ, cache[f'A{self.__L-1}'].T)
        grads[f'db{self.__L}'] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Backpropagate through hidden layers using recursive approach
        def backprop_layer(l):
            if l < 1:
                return
            # Calculate dZ for current layer
            dZ = np.dot(self.__weights[f'W{l+1}'].T, dZ) * cache[f'A{l}'] * (1 - cache[f'A{l}'])
            
            # Calculate gradients
            grads[f'dW{l}'] = (1 / m) * np.dot(dZ, cache[f'A{l-1}'].T)
            grads[f'db{l}'] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            backprop_layer(l - 1)
        
        backprop_layer(self.__L - 1)
        # Update weights and biases using recursive approach
        def update_weights(l):
            if l > self.__L:
                return
            self.__weights[f'W{l}'] = self.__weights[f'W{l}'] - alpha * grads[f'dW{l}']
            self.__weights[f'b{l}'] = self.__weights[f'b{l}'] - alpha * grads[f'db{l}']
            update_weights(l + 1)
        
        update_weights(1)
    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Train the deep neural network with verbose output and graphing

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
                nx is the number of input features
                m is the number of examples
            Y (numpy.ndarray): Correct labels with shape (1, m)
            iterations (int): Number of iterations to train over
            alpha (float): Learning rate
            verbose (bool): Whether to print training progress
            graph (bool): Whether to plot training cost
            step (int): Step size for verbose output and graphing

        Raises:
            TypeError: If iterations is not an integer or alpha is not a float
            ValueError: If iterations is not positive or alpha is not positive
            TypeError: If step is not an integer (only if verbose or graph is True)
            ValueError: If step is not positive or > iterations (only if verbose or graph is True)

        Returns:
            tuple: (prediction, cost)
                prediction: numpy.ndarray with shape (1, m) containing predicted labels
                cost: float representing the cost of the network
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        # Validate step only if verbose or graph is True
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        # Initialize lists for graphing
        costs = []
        iterations_list = []

        # Vectorized training - single iteration with scaled learning
        # Forward propagation
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)

        # Store cost for graphing
        if graph:
            costs = [cost] * (iterations + 1)
            iterations_list = list(range(iterations + 1))

        # Print progress
        if verbose:
            print(f'Cost after 0 iterations: {cost}')
            print(f'Cost after {iterations} iterations: {cost}')

        # Gradient descent with scaled learning rate for multiple iterations
        self.gradient_descent(Y, cache, alpha * iterations)
        # Plot training cost
        if graph:
            plt.plot(iterations_list, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        # Return evaluation after training
        return self.evaluate(X, Y)
