# Classification

This directory contains implementations for binary classification using neural networks.

## Files

### 0-neuron.py
A single neuron implementation for binary classification.

#### Neuron Class

The `Neuron` class represents a single neuron that can perform binary classification.

**Constructor:**
```python
def __init__(self, nx):
```

**Parameters:**
- `nx` (int): Number of input features to the neuron

**Raises:**
- `TypeError`: If `nx` is not an integer
- `ValueError`: If `nx` is less than 1

**Public Attributes:**
- `W`: Weights vector initialized with random normal distribution (shape: 1 × nx)
- `b`: Bias initialized to 0
- `A`: Activated output initialized to 0

**Example Usage:**
```python
import numpy as np
from 0-neuron import Neuron

# Create a neuron with 784 input features
neuron = Neuron(784)

# Access the attributes
print(f"Weights shape: {neuron.W.shape}")  # (1, 784)
print(f"Bias: {neuron.b}")                 # 0
print(f"Activated output: {neuron.A}")     # 0

# Modify the activated output
neuron.A = 10
print(f"New activated output: {neuron.A}") # 10
```

**Error Handling:**
```python
# This will raise TypeError
neuron = Neuron("invalid")

# This will raise ValueError
neuron = Neuron(0)
```

## Requirements

- Python 3.x
- NumPy

## Installation

Make sure you have NumPy installed:
```bash
pip install numpy
```
