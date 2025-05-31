#bin

def determinant(matrix):
    """Calculates the determinant of a matrix"""
    # Validate input is a list of lists
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    
    # Check for 0x0 matrix [[]]
    if matrix == [[]]:
        return 1

    # Check for square matrix
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base case: 1x1 matrix
    if size == 1:
        return matrix[0][0]

    # Base case: 2x2 matrix
    if size == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    # Recursive case: expand using first row
    det = 0
    for col in range(size):
        # Build minor matrix
        minor = [
            [matrix[i][j] for j in range(size) if j != col]
            for i in range(1, size)
        ]
        # Alternate signs and recurse
        sign = (-1) ** col
        det += sign * matrix[0][col] * determinant(minor)

    return det
