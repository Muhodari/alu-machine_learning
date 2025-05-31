#!/usr/bin/env python3
"""
Module that calculates the adjugate matrix of a square matrix.
"""


def determinant(matrix):
    """Calculates the determinant of a square matrix."""
    if matrix == [[]]:
        return 1
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return (matrix[0][0] * matrix[1][1] -
                matrix[0][1] * matrix[1][0])

    det = 0
    for col in range(len(matrix)):
        sub = []
        for i in range(1, len(matrix)):
            row = []
            for j in range(len(matrix)):
                if j != col:
                    row.append(matrix[i][j])
            sub.append(row)
        det += ((-1) ** col) * matrix[0][col] * determinant(sub)
    return det


def cofactor(matrix):
    """Calculates the cofactor matrix of a square matrix."""
    if (not isinstance(matrix, list) or
            not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")
    if (matrix == [] or any(len(row) != len(matrix) for row in matrix)):
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]

    result = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix)):
            minor = []
            for x in range(len(matrix)):
                if x != i:
                    minor_row = []
                    for y in range(len(matrix)):
                        if y != j:
                            minor_row.append(matrix[x][y])
                    minor.append(minor_row)
            value = ((-1) ** (i + j)) * determinant(minor)
            row.append(value)
        result.append(row)
    return result


def adjugate(matrix):
    """
    Calculates the adjugate matrix of a square matrix.

    Args:
        matrix: List of lists representing a square matrix.

    Returns:
        List of lists representing the adjugate matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a non-empty square matrix.
    """
    if (not isinstance(matrix, list) or
            not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")
    if (matrix == [] or any(len(row) != len(matrix) for row in matrix)):
        raise ValueError("matrix must be a non-empty square matrix")

    cof = cofactor(matrix)
    # Transpose the cofactor matrix
    adj = []
    for i in range(len(cof)):
        row = []
        for j in range(len(cof)):
            row.append(cof[j][i])
        adj.append(row)
    return adj
