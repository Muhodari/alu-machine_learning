#!/usr/bin/env python3
"""
This module provides a function to perform same convolution on grayscale
images.
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.
    images: numpy.ndarray of shape (m, h, w) containing multiple grayscale
        images
    kernel: numpy.ndarray of shape (kh, kw) containing the kernel for the
        convolution
    Returns: numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    pad_h = (kh - 1) // 2 if kh % 2 == 1 else kh // 2
    pad_w = (kw - 1) // 2 if kw % 2 == 1 else kw // 2
    padded = np.pad(
        images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant'
    )
    output = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            # Use broadcasting to apply the kernel to all images at once
            output[:, i, j] = np.sum(
                padded[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2)
            )
    return output
