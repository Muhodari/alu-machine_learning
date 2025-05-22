#!/usr/bin/env python3
"""
This module provides a function to add two arrays element-wise.
"""


def add_arrays(arr1, arr2):
    """
    Adds two arrays element-wise.

    Args:
        arr1 (list): The first list of numbers.
        arr2 (list): The second list of numbers.

    Returns:
        list or None: A new list with element-wise sums, or None if shapes differ.
    """
    if len(arr1) != len(arr2):
        return None
    return [a + b for a, b in zip(arr1, arr2)]
