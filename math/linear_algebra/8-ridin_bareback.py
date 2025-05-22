#!/usr/bin/env python3
"""Module to perform matrix multiplication."""


def mat_mul(mat1, mat2):
    """Performs matrix multiplication on two 2D matrices."""
    if len(mat1[0]) != len(mat2):
        return None

    result = []
    for row in mat1:
        new_row = []
        for col in zip(*mat2):
            product = sum(a * b for a, b in zip(row, col))
            new_row.append(product)
        result.append(new_row)

    return result
