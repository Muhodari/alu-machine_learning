#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Time range
x = np.arange(0, 21000, 1000)

# Decay constant
r = np.log(0.5)

# Half-lives
t1 = 5730   # C-14
t2 = 1600   # Ra-226

# Exponential decay calculations
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

# Plotting
plt.plot(x, y1, 'r--', label='C-14')      # Red dashed line
plt.plot(x, y2, 'g-', label='Ra-226')     # Green solid line

# Labels and title
plt.title('Exponential Decay of Radioactive Elements')
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')

# Add legend
plt.legend()

# Show plot
plt.show()
