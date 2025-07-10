#!/usr/bin/env python3
"""
This module provides a function to perform convolution on grayscale images
with custom padding.
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.
    images: numpy.ndarray of shape (m, h, w) containing multiple grayscale
        images
    kernel: numpy.ndarray of shape (kh, kw) containing the kernel for the
        convolution
    padding: tuple of (ph, pw)
        ph: padding for the height
        pw: padding for the width
    Returns: numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    padded = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw)), mode='constant'
    )
    output_h = h + 2 * ph - kh + 1
    output_w = w + 2 * pw - kw + 1
    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            # Use broadcasting to apply the kernel to all images at once
            output[:, i, j] = np.sum(
                padded[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2)
            )
    return output
