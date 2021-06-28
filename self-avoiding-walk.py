#!/usr/bin/env python
# coding: utf-8

# Refer to the [Wikipedia article](https://en.wikipedia.org/wiki/Self-avoiding_walk) on self-avoiding walks for a good primer on the subject. [Bauerschmidt et al. (2012)](https://www.ihes.fr/~duminil/publi/saw_lecture_notes.pdf) give an extremely thorough description of known qualities of self-avoiding random walks and their connections to other areas of mathematics. Here are links to some other resources I found informative:
#  - https://mathoverflow.net/questions/158811/wander-distance-of-self-avoiding-walk-that-backs-out-of-culs-de-sac
#  - https://mathoverflow.net/questions/52813/self-avoiding-walk-enumerations
#  - https://mathoverflow.net/questions/41543/how-to-characterize-a-self-avoiding-
#  - https://mathoverflow.net/questions/54144/self-avoiding-walk-pair-correlation
#  - https://mathoverflow.net/questions/23583/self-avoidance-time-of-random-walk
#  - https://mathoverflow.net/questions/181340/square-filling-self-avoiding-walk

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
@nb.jit(nopython=True)
def bound(x, a, b):
    if x >= b:
        x = b-1
    elif x < a:
        x = a
    return x

@nb.njit
def clip(x, a, b):
    for i in range(x.shape[0]):
        x[i] = bound(x[i], a, b)
    return x
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




