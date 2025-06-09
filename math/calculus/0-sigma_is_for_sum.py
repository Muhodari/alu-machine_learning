#!/usr/bin/env python3
"""
This script computes  the summation of integers from i=2 to i=5.
"""

def summation(start, end):
    """Returns the sum from start to end inclusive."""
    return sum(range(start, end + 1))

if __name__ == "__main__":
    print(summation(2, 5))  # Output: 14
