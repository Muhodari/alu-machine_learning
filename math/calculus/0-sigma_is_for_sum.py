#!/usr/bin/env python3
"""
This module provides a function that calculates the summation of
integers from a given starting integer i up to a given ending integer n.
"""


def summation_i_to_n(i=2, n=5):
    """
    Calculates the sum of integers from i to n (inclusive).

    Args:
        i (int): Starting integer (default is 2)
        n (int): Ending integer (default is 5)

    Returns:
        int: The sum of integers from i to n
    """
    return sum(range(i, n + 1))


if __name__ == "__main__":
    print(summation_i_to_n())
