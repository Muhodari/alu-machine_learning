#!/usr/bin/env python3
"""
This module provides a function to concatenate two matrices
along a specified axis without using NumPy.
"""

def cat_matrices(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis."""
    if axis == 0:
        if any(len(row) != len(mat1[0]) for row in mat1 + mat2):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]

    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]

    else:
        # Recursive handling for axis > 1
        if len(mat1) != len(mat2):
            return None
        result = []
        for sub1, sub2 in zip(mat1, mat2):
            cat = cat_matrices(sub1, sub2, axis=axis - 1)
            if cat is None:
                return None
            result.append(cat)
        return result
