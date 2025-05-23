#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Generate grades
np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# Plotting the histogram
plt.hist(student_grades, bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
         edgecolor='black')

# Adding titles and labels
plt.title('Project A')
plt.xlabel('Grades')
plt.ylabel('Number of Students')

# Display the plot
plt.show()
