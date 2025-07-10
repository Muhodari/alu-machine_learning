#!/usr/bin/env python3
"""
This module provides a function to perform convolution on images with channels
with support for custom padding and stride.
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels.
    images: numpy.ndarray of shape (m, h, w, c) containing multiple images
    kernel: numpy.ndarray of shape (kh, kw, c) containing the kernel for the
        convolution
    padding: tuple of (ph, pw), 'same', or 'valid'
    stride: tuple of (sh, sw)
    Returns: numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    if type(padding) is tuple:
        ph, pw = padding
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + ((h - 1) * sh + kh - h) % 2
        pw = ((w - 1) * sw + kw - w) // 2 + ((w - 1) * sw + kw - w) % 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        raise ValueError("padding must be a tuple, 'same', or 'valid'")

    padded = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant'
    )
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            # Use broadcasting to apply the kernel to all images at once
            output[:, i, j] = np.sum(
                padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :] * kernel,
                axis=(1, 2, 3)
            )
    return output
