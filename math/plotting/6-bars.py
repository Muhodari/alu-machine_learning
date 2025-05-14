#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

persons = ['Farrah', 'Fred', 'Felicia']
apples = fruit[0]
bananas = fruit[1]
oranges = fruit[2]
peaches = fruit[3]

plt.bar(persons, apples, color='red', label='apples')
plt.bar(persons, bananas, bottom=apples, color='yellow', label='bananas')
plt.bar(persons, oranges, bottom=apples + bananas, color='orange', label='oranges')
plt.bar(persons, peaches, bottom=apples + bananas + oranges, color='peachpuff', label='peaches')

plt.ylabel("Quantity of Fruit")
plt.title("Number of Fruit per Person")
plt.legend()
plt.ylim(0, 80)

plt.show()
