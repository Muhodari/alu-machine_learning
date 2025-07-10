# Convolutions and Pooling

This directory contains scripts related to convolution and pooling operations, commonly used in image processing and deep learning.

## Scripts

### 0-convolve_grayscale_valid.py
- **Function:** `convolve_grayscale_valid(images, kernel)`
- **Description:** Performs a valid convolution on a batch of grayscale images using a specified kernel.

### 1-convolve_grayscale_same.py
- **Function:** `convolve_grayscale_same(images, kernel)`
- **Description:** Performs a same convolution (output size matches input) on grayscale images, using zero-padding as needed.

### 2-convolve_grayscale_padding.py
- **Function:** `convolve_grayscale_padding(images, kernel, padding)`
- **Description:** Performs a convolution on grayscale images with custom zero padding.

### 3-convolve_grayscale.py
- **Function:** `convolve_grayscale(images, kernel, padding='same', stride=(1, 1))`
- **Description:** Performs a convolution on grayscale images with support for 'same', 'valid', or custom tuple padding, and arbitrary stride.

### 4-convolve_channels.py
- **Function:** `convolve_channels(images, kernel, padding='same', stride=(1, 1))`
- **Description:** Performs a convolution on images with channels (e.g., RGB), supporting 'same', 'valid', or custom tuple padding, and arbitrary stride.

### 5-convolve.py
- **Function:** `convolve(images, kernels, padding='same', stride=(1, 1))`
- **Description:** Performs a convolution on images using multiple kernels, supporting 'same', 'valid', or custom tuple padding, and arbitrary stride.

### 6-pool.py
- **Function:** `pool(images, kernel_shape, stride, mode='max')`
- **Description:** Performs pooling (max or average) on images using the specified kernel shape and stride.

---

Add more scripts and documentation as you implement additional convolution and pooling operations. 