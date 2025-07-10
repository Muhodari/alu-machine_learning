# Convolutions and Pooling

This directory contains scripts related to convolution and pooling operations, commonly used in image processing and deep learning.

## Scripts

### 0-convolve_grayscale_valid.py
- **Function:** `convolve_grayscale_valid(images, kernel)`
- **Description:** Performs a valid convolution on a batch of grayscale images using a specified kernel.
- **Parameters:**
  - `images`: numpy.ndarray of shape (m, h, w) containing multiple grayscale images
    - `m`: number of images
    - `h`: height in pixels
    - `w`: width in pixels
  - `kernel`: numpy.ndarray of shape (kh, kw) containing the kernel for the convolution
    - `kh`: height of the kernel
    - `kw`: width of the kernel
- **Returns:** numpy.ndarray containing the convolved images

### Example Usage
```python
import numpy as np
from 0-convolve_grayscale_valid import convolve_grayscale_valid

images = np.random.rand(10, 28, 28)  # 10 random grayscale images
kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
output = convolve_grayscale_valid(images, kernel)
print(output.shape)  # (10, 26, 26)
```

---

Add more scripts and documentation as you implement additional convolution and pooling operations. 