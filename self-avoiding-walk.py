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


# In[666]:


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
choices = np.stack(choices)

print(choices)


# In[650]:


steps = []
@nb.njit
def valid_moves(g, m, q):
#     filtered = list(filter(lambda c: (0<=pos+c).all() and (pos+c<z).all() and grid[tuple(pos+c)] == 0, m))
    filtered = []
    for i in m:
#         print(pos, m)
        p = q+i
#         if (0<=p).all() and (p<z).all() and g[p[0], p[1]] == 0:
        if (0<=p).all():
            if (p<z).all():
                if g[p[0], p[1]] == 0:
#                     print(p, g[p[0], p[1]], (p<z).all(), z)
                    filtered.append(i)
    return filtered


# In[667]:


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


# In[690]:


@nb.njit#(parallel=True)
def simulate():
    for x in range(1):
        pos = np.array([4,4])
#         grid = np.zeros([z] * D)
        grid = np.zeros((z, z), dtype=np.int64)
        lengths = []
        walks = []
        for t in range(z**2):
    #         print(0<pos+delta[0]<z)
    #         print(grid[tuple(pos+delta[0])])
            possible = valid_moves(grid, choices, pos)
#             print(possible)
            grid[pos[0], pos[1]] = t+(z**2//4)
            
            if len(possible) > 0:
#                 delta = random.choice(possible)
#                 delta = np.random.choice(possible)
#                 np.random.shuffle(possible)
                index = np.random.randint(0, len(possible))
                delta = possible[index]

#                 steps.append(delta)
                pos += delta
#                 pos = np.clip(pos, 0, z-1)
                
                pos = clip(pos, 0, z)
                
#                 grid[tuple(pos)] = 1
#                 print(pos[0])
            else:
                lengths.append(t)
#                 walks.append(grid)
                break
#         else:
        walks.append(grid)
    return grid


# In[732]:


best = None
lengths = []
walks = []
for i in range(2000):
    G = simulate()
#     if best:
#         print(best.max())
    lengths.append(G.max())
    walks.append(G)
    if best is None or G.max() > best.max():
        best = G

plt.figure(figsize=(10, 10))
plt.imshow(best, cmap='inferno')
plt.axis('off')
# decision trees? + parity
# random walks that close to a polygon
# (self-avoiding) random walks around obstructions
# add heuristics
# avoid and/or break at 
# add backtracking


# In[685]:


plt.hist(lengths, bins=30)


# In[619]:


plt.imshow(np.average(np.stack(walks), axis=0))


# In[ ]:




