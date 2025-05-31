#!/usr/bin/env python3
"""Module that provides a function to calculate the minor matrix of a matrix."""


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
        minor = []
        for i in range(1, size):
            row = []
            for j in range(size):
                if j != col:
                    row.append(matrix[i][j])
            minor.append(row)
        sign = (-1) ** col
        det += sign * matrix[0][col] * determinant(minor)
    return det


def minor(matrix):
    """Calculate the minor matrix of a square matrix.

    Args:
        matrix (list of lists): A non-empty square matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not square or is empty.

    Returns:
        list of lists: The minor matrix.
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

    minors = []
    for i in range(size):
        row_minors = []
        for j in range(size):
            submatrix = [
                [matrix[x][y] for y in range(size) if y != j]
                for x in range(size) if x != i
            ]
            row_minors.append(determinant(submatrix))
        minors.append(row_minors)
    return minors
