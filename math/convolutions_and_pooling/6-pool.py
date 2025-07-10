#!/usr/bin/env python3
"""
This module provides a function to perform pooling (max or average) on images.
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.
    images: numpy.ndarray of shape (m, h, w, c) containing multiple images
    kernel_shape: tuple of (kh, kw) containing the kernel shape for pooling
    stride: tuple of (sh, sw)
    mode: 'max' or 'avg' for max or average pooling
    Returns: numpy.ndarray containing the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1
    output = np.zeros((m, out_h, out_w, c))

    for i in range(out_h):
        for j in range(out_w):
            window = images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            if mode == 'max':
                output[:, i, j, :] = np.max(window, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(window, axis=(1, 2))
            else:
                raise ValueError("mode must be 'max' or 'avg'")
    return output
