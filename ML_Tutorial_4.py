#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
from numpy.random import normal
from scipy.stats import norm, multivariate_normal
from math import sqrt


# In[2]:


x = np.linspace(-1, 1, 10)

def line(x, w0=-0.3, w1=0.5):
    return w0 + w1*x

y,t = [],[]
for i in x:
    i = line(i)
    t.append(i)
    y.append(i+normal(0,0.2))
plt.scatter(x,y)
plt.grid()
plt.plot(x, t, "red")
plt.show()


# In[3]:


def gaussian(x, y, w0, w1):
    dev = 0.2
    const = sqrt(np.pi * 2) * dev
    return const * np.exp(-0.5 * ((y - line(x, w0, w1))/dev) ** 2)

def post(s0, m0, phi, beta, t):
    s0_inv = np.linalg.inv(s0)
    Sn = np.linalg.inv(s0_inv + beta * phi.T@phi)
    M = s0_inv@(m0.reshape((2, 1))) + beta * phi.T * t
    Mn = Sn @ M 
    return Mn.reshape((1, 2))[0], Sn

w0, w1 = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
p0 = norm.pdf(w0, 0, sqrt(0.5))
p1 = norm.pdf(w1, 0, sqrt(0.5))
prior = p0 * p1
m0, s0 = np.array([0, 0]), np.array([[1, 0], [0, 1]])
beta = 25
fig, axs = plt.subplots(10, 3, figsize=(12, 30))
count = 0
new_x, new_y = [], []
for i, j in zip(x, y):
    likelihood = gaussian(i, j, w0, w1)
    posterior = prior * likelihood
    prior = posterior
    phi = np.array([1,i]).reshape((1,2))
    m0, s0 = post(s0, m0, phi, beta, j)
    d = multivariate_normal.rvs(m0, s0, 10)
    new_x.append(i)
    new_y.append(j)
    axs[count, 0].contourf(w0, w1, likelihood, cmap="jet")
    axs[count, 1].contourf(w0, w1, posterior, cmap="jet")
    for l in d:
      axs[count, 2].plot(x, l[0] + l[1] * x, c="red")
    axs[count, 2].scatter(new_x, new_y, s=80, facecolors='none', edgecolors='blue')
    count += 1
fig.tight_layout()


# In[ ]:





# In[ ]:




