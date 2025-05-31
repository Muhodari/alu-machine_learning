#!/usr/bin/env python3
"""
Module that provides a function to calculate
the cofactor matrix of a square matrix.
"""


def determinant(matrix):
    """Calculate the determinant of a square matrix."""
    if matrix == [[]]:
        return 1
    size = len(matrix)
    if size == 1:
        return matrix[0][0]
    if size == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for col in range(size):
        sub = []
        for i in range(1, size):
            row = []
            for j in range(size):
                if j != col:
                    row.append(matrix[i][j])
            sub.append(row)
        det += ((-1) ** col) * matrix[0][col] * determinant(sub)
    return det


def cofactor(matrix):
    """
    Calculate the cofactor matrix of a square matrix.

    Args:
        matrix (list of lists): A non-empty square matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not square or is empty.

    Returns:
        list of lists: The cofactor matrix.
    """
    if not isinstance(matrix, list) or not all(
        isinstance(row, list) for row in matrix
    ):
        raise TypeError("matrix must be a list of lists")
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    size = len(matrix)
    if size == 1:
        return [[1]]

    cofactors = []
    for i in range(size):
        row_cofactors = []
        for j in range(size):
            sub = [
                [matrix[r][c] for c in range(size) if c != j]
                for r in range(size) if r != i
            ]
            cofactor_val = ((-1) ** (i + j)) * determinant(sub)
            row_cofactors.append(cofactor_val)
        cofactors.append(row_cofactors)
    return cofactors
