#!/usr/bin/env python3
"""Add two matrices of any shape recursively"""


def add_matrices(mat1, mat2):
    """Adds two matrices recursively"""
    if type(mat1) != type(mat2):
        return None

    if isinstance(mat1, list) and isinstance(mat2, list):
        if len(mat1) != len(mat2):
            return None
        result = []
        for a, b in zip(mat1, mat2):
            added = add_matrices(a, b)
            if added is None:
                return None
            result.append(added)
        return result

    try:
        return mat1 + mat2
    except TypeError:
        return None
