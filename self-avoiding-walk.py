#!/usr/bin/env python
# coding: utf-8

# Refer to the [Wikipedia article](https://en.wikipedia.org/wiki/Self-avoiding_walk) on self-avoiding walks for a good primer on the subject.

# In[32]:


import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import itertools
import random


# In[244]:


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
