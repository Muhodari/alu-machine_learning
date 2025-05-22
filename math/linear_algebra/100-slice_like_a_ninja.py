#!/usr/bin/env python3
"""Slice a matrix along specific axes"""
import numpy as np


def np_slice(matrix, axes={}):
    """Slices a matrix along specific axes based on the axes dictionary"""
    # Build the list of slices
    slicer = [slice(None)] * matrix.ndim  # full slice for each axis
    for axis, slc in axes.items():
        slicer[axis] = slice(*slc)  # unpack the tuple (start, stop, step)
    return matrix[tuple(slicer)]
