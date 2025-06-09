#!/usr/bin/env python3
"""
This module provides a function to evaluate the summation
∑(9i - 2k) from i = 1 to 4.
"""


def summation_expression(k):
    """
    Calculates the value of ∑(9i - 2k) for i = 1 to 4.

    Args:
        k (int): The constant multiplier used in the expression

    Returns:
        int: The evaluated result of the summation
    """
    return sum(9 * i - 2 * k for i in range(1, 5))


if __name__ == "__main__":
    k = 1  # You can test with other values
    print(summation_expression(k))
