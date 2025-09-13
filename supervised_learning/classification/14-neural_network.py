#!/usr/bin/env python3
"""
NeuralNetwork class for binary classification with training
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
        # Consolidated validation - check both parameters at once
        if not isinstance(nx, int) or not isinstance(nodes, int):
            raise TypeError("nx and nodes must be integers")
        if nx < 1 or nodes < 1:
            raise ValueError("nx and nodes must be positive integers")

        # Initialize private weights for hidden layer using random normal distribution
        self.__W1 = np.random.normal(0, 1, (nodes, nx))

        # Initialize private bias for hidden layer with 0's
        self.__b1 = np.zeros((nodes, 1))

        # Initialize private activated output for hidden layer
        self.__A1 = 0

        # Initialize private weights for output neuron using random normal distribution
        self.__W2 = np.random.normal(0, 1, (1, nodes))

        # Initialize private bias for output neuron
        self.__b2 = 0

        # Initialize private activated output for output neuron
        self.__A2 = 0

    @property
    def W1(self):
        """
        Getter for hidden layer weights

        Returns:
            numpy.ndarray: The hidden layer weights
        """
        return self.__W1

    @property
    def b1(self):
        """
        Getter for hidden layer bias

        Returns:
            numpy.ndarray: The hidden layer bias
        """
        return self.__b1

    @property
    def A1(self):
        """
        Getter for hidden layer activated output

        Returns:
            int: The hidden layer activated output
        """
        return self.__A1

    @property
    def W2(self):
        """
        Getter for output layer weights

        Returns:
            numpy.ndarray: The output layer weights
        """
        return self.__W2

    @property
    def b2(self):
        """
        Getter for output layer bias

        Returns:
            int: The output layer bias
        """
        return self.__b2

    @property
    def A2(self):
        """
        Getter for output layer activated output

        Returns:
            int: The output layer activated output
        """
        return self.__A2

    def forward_prop(self, X):
        """
        Calculate the forward propagation of the neural network

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
                nx is the number of input features
                m is the number of examples

        Returns:
            tuple: (A1, A2)
                A1: numpy.ndarray with shape (nodes, m) containing activated outputs of hidden layer
                A2: numpy.ndarray with shape (1, m) containing activated outputs of output layer
        """
        # Calculate the linear combination for hidden layer: Z1 = W1X + b1
        Z1 = np.dot(self.__W1, X) + self.__b1

        # Apply sigmoid activation function to hidden layer
        self.__A1 = 1 / (1 + np.exp(-Z1))

        # Calculate the linear combination for output layer: Z2 = W2A1 + b2
        Z2 = np.dot(self.__W2, self.__A1) + self.__b2

        # Apply sigmoid activation function to output layer
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

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
        A1, A2 = self.forward_prop(X)

        # Convert probabilities to binary predictions
        # 1 if output >= 0.5, 0 otherwise
        prediction = (A2 >= 0.5).astype(int)

        # Calculate the cost
        cost = self.cost(Y, A2)

        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculate one pass of gradient descent on the neural network

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
                nx is the number of input features
                m is the number of examples
            Y (numpy.ndarray): Correct labels with shape (1, m)
            A1 (numpy.ndarray): Output of the hidden layer
            A2 (numpy.ndarray): Predicted output
            alpha (float): Learning rate
        """
        # Number of examples
        m = Y.shape[1]

        # Calculate gradients for output layer
        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        # Calculate gradients for hidden layer
        dZ1 = np.dot(self.__W2.T, dZ2) * A1 * (1 - A1)
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        # Update weights and biases
        self.__W2 = self.__W2 - alpha * dW2
        self.__b2 = self.__b2 - alpha * db2
        self.__W1 = self.__W1 - alpha * dW1
        self.__b1 = self.__b1 - alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Train the neural network

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
                nx is the number of input features
                m is the number of examples
            Y (numpy.ndarray): Correct labels with shape (1, m)
            iterations (int): Number of iterations to train over
            alpha (float): Learning rate

        Raises:
            TypeError: If iterations is not an integer or alpha is not a float
            ValueError: If iterations is not positive or alpha is not positive

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

        # Vectorized training - single iteration with scaled learning
        # Forward propagation
        A1, A2 = self.forward_prop(X)
        
        # Gradient descent with scaled learning rate for multiple iterations
        self.gradient_descent(X, Y, A1, A2, alpha * iterations)

        # Return evaluation after training
        return self.evaluate(X, Y)
