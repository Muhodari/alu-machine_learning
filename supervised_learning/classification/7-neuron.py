#!/usr/bin/env python3
"""
Neuron class for binary classification with verbose and graph training
"""

import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """
    A single neuron performing binary classification
    """

    def __init__(self, nx):
        """
        Initialize a neuron

        Args:
            nx (int): Number of input features to the neuron

        Raises:
            TypeError: If nx is not an integer
            ValueError: If nx is less than 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        # Initialize private weights using random normal distribution
        self.__W = np.random.normal(0, 1, (1, nx))

        # Initialize private bias to 0
        self.__b = 0

        # Initialize private activated output to 0
        self.__A = 0

    @property
    def W(self):
        """
        Getter for weights vector

        Returns:
            numpy.ndarray: The weights vector
        """
        return self.__W

    @property
    def b(self):
        """
        Getter for bias

        Returns:
            int: The bias value
        """
        return self.__b

    @property
    def A(self):
        """
        Getter for activated output

        Returns:
            int: The activated output value
        """
        return self.__A

    def forward_prop(self, X):
        """
        Calculate the forward propagation of the neuron

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
                nx is the number of input features
                m is the number of examples

        Returns:
            numpy.ndarray: The activated output (__A)
        """
        # Calculate the linear combination: Z = WX + b
        Z = np.dot(self.__W, X) + self.__b

        # Apply sigmoid activation function
        self.__A = 1 / (1 + np.exp(-Z))

        return self.__A

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
        Evaluate the neuron's predictions

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
        A = self.forward_prop(X)

        # Convert probabilities to binary predictions
        # 1 if output >= 0.5, 0 otherwise
        prediction = (A >= 0.5).astype(int)

        # Calculate the cost
        cost = self.cost(Y, A)

        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculate one pass of gradient descent on the neuron

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
                nx is the number of input features
                m is the number of examples
            Y (numpy.ndarray): Correct labels with shape (1, m)
            A (numpy.ndarray): Activated output with shape (1, m)
            alpha (float): Learning rate (default: 0.05)
        """
        # Number of examples
        m = Y.shape[1]

        # Calculate the gradient of the cost with respect to weights
        # dW = (1/m) * X * (A - Y).T
        dW = (1 / m) * np.dot(X, (A - Y).T)

        # Calculate the gradient of the cost with respect to bias
        # db = (1/m) * sum(A - Y)
        db = (1 / m) * np.sum(A - Y)

        # Update weights: W = W - alpha * dW
        self.__W = self.__W - alpha * dW.T

        # Update bias: b = b - alpha * db
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Train the neuron with verbose output and graphing

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

        # Training loop
        for i in range(iterations + 1):
            # Forward propagation
            A = self.forward_prop(X)
            cost = self.cost(Y, A)

            # Store cost for graphing
            if graph:
                costs.append(cost)
                iterations_list.append(i)

            # Print progress
            if verbose and (i % step == 0 or i == 0 or i == iterations):
                print(f"Cost after {i} iterations: {cost}")

            # Gradient descent (skip on iteration 0)
            if i < iterations:
                self.gradient_descent(X, Y, A, alpha)

        # Plot training cost
        if graph:
            plt.plot(iterations_list, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        # Return evaluation after training
        return self.evaluate(X, Y)
