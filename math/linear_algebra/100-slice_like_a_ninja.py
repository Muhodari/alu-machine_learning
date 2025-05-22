#!/usr/bin/env python3
"""Slice a matrix along specific axes"""


def np_slice(matrix, axes={}):
    """Slices a matrix along specific axes without importing numpy"""
    slicer = [slice(None)] * matrix.ndim
    for axis, s in axes.items():
        slicer[axis] = slice(*s)
    return matrix[tuple(slicer)]

