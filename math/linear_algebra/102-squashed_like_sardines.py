#!/usr/bin/env python3

def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis.

    Parameters:
    - mat1: First matrix (list of ints/floats or nested lists)
    - mat2: Second matrix (same type/shape as mat1)
    - axis: Axis along which to concatenate (default is 0)

    Returns:
    - A new matrix after concatenation, or None if the matrices can't be concatenated
    """
    # Helper function to get the shape of a matrix
    def get_shape(matrix):
        shape = []
        while isinstance(matrix, list):
            shape.append(len(matrix))
            if not matrix:  # Handle empty sublist
                break
            matrix = matrix[0]
        return shape

    # Helper function to recursively concatenate two matrices at a specific axis
    def recursive_concat(m1, m2, axis):
        if axis == 0:
            return m1 + m2
        if len(m1) != len(m2):
            return None
        result = []
        for sub1, sub2 in zip(m1, m2):
            merged = recursive_concat(sub1, sub2, axis - 1)
            if merged is None:
                return None
            result.append(merged)
        return result

    shape1 = get_shape(mat1)
    shape2 = get_shape(mat2)

    if len(shape1) != len(shape2):
        return None

    for i in range(len(shape1)):
        if i != axis and shape1[i] != shape2[i]:
            return None

    return recursive_concat(mat1, mat2, axis)
