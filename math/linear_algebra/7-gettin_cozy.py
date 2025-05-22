#!/usr/bin/env python3
"""Module to concatenate two 2D matrices along a specific axis."""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates two 2D matrices along the specified axis."""
    if axis == 0:
        if not all(len(row) == len(mat1[0]) for row in mat2):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]

    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [
            row1[:] + row2[:] for row1, row2 in zip(mat1, mat2)
        ]

    return None
