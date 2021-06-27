#!/usr/bin/env python
# coding: utf-8

# Refer to the [Wikipedia article](https://en.wikipedia.org/wiki/Self-avoiding_walk) on self-avoiding walks for a good primer on the subject.

# In[32]:


import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import itertools
import random


# In[264]:


dimensions = 2
D = dimensions
z = 10

start = [0, 0]
choices = []

for n in range(dimensions):
    for y in [-1, 1]:
        delta = np.zeros(dimensions).astype(np.int)
        delta[n] = y
        choices.append(delta)

steps = []
pos = np.array(start)
lengths = []
walks = []
for x in range(1000):
    grid = np.zeros([z] * D)
    for t in range(z**2):
#         print(0<pos+delta[0]<z)
#         print(grid[tuple(pos+delta[0])])
        possible = list(filter(lambda c: (0<=pos+c).all() and (pos+c<z).all() and grid[tuple(pos+c)] == 0, choices))
        if possible:
            delta = random.choice(possible)
            steps.append(delta)
            pos += delta
            pos = np.clip(pos, 0, z-1)
            grid[tuple(pos)] = 1
        else:
            lengths.append(t)
            walks.append(grid)
            break

plt.imshow(grid, cmap='inferno')


# In[265]:


plt.imshow(np.average(np.stack(walks), axis=0))


# In[263]:


plt.hist(lengths, bins=25)


# In[ ]:




