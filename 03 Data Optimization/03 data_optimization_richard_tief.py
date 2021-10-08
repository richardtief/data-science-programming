#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# In[2]:


# read the csv file and get basic info
df = pd.read_csv('cost.csv')
df.info()
df


# In[3]:


# convert to numpy array to ensure integer
# costs = df.to_numpy(dtype=int)
costs = np.loadtxt("cost.csv", delimiter=",", dtype=int)

# point with minimum costs
minimum = sorted([np.where(costs == np.amin(costs))[i][0] for i in range(2)])
print(minimum)
# point with maximum costs
maximum = sorted([np.where(costs == np.amax(costs))[i][0] for i in range(2)])
print(maximum)


# In[4]:


# explore the data
plt.rcParams["figure.figsize"] = (20,10)
plt.contourf(df, levels=255, cmap='terrain_r')
plt.plot(*minimum, 'rX', color="blue", markersize=12)
plt.plot(*maximum, 'rX', markersize=12)
plt.colorbar()
plt.show()


# In[5]:


# cost function with error handling
def cost_func(x, y):
    if x < 0 or y < 0:
        return np.amax(costs)+1 # return high costs
    try:
        return costs[x, y]
    except IndexError:
        return np.amax(costs)+1 # return high costs


# In[6]:


def nelder_mead(data, alpha, sigma, gamma, rho, t, points):

    # creating numpy array
    dt = np.dtype(dtype=[('point', int, 2), ('cost', int)])
    simplex = np.array(points, dtype=dt)
    # add costs to every point in the simplex
    simplex['cost'] = [cost_func(*point) for point in simplex['point']]

    while True:
        # sort points by cost
        simplex = np.sort(simplex, order="cost")
#         print(f'new simplex:\n{simplex[0]}\n{simplex[1]}\n{simplex[2]}')

        x_best = simplex['point'][0]
        x_second_worst = simplex['point'][-2]
        x_worst = simplex['point'][-1]

        # calculate the centroid of the simplex. The centroid does not include the worst point
        x_centroid = np.mean(simplex['point'][:2], dtype=int)

        # calculated points
        x_reflected = alpha * (x_centroid - x_worst) + x_centroid
        x_expanded = x_centroid + (gamma * x_reflected - x_centroid)
        x_contracted = x_centroid + (rho * (x_worst - x_centroid)).astype(int)

        if np.std(simplex['cost']) < t and (np.std(simplex['point'], axis=0) == 0).all():
            break

        # reflection
        if cost_func(*x_best) <= cost_func(*x_reflected) < cost_func(*x_second_worst):
            simplex[-1] = x_reflected, cost_func(*x_reflected)
            print(f'Reflected to {x_reflected} with costs {cost_func(*x_reflected)}')

        # reflection expansion
        elif cost_func(*x_reflected) < cost_func(*x_best):
            if cost_func(*x_expanded) < cost_func(*x_reflected):
                simplex[-1] = x_expanded, cost_func(*x_expanded)
                print(f'Expanded to {x_reflected} with costs {cost_func(*x_reflected)}')

            else:
                simplex[-1] = x_reflected, cost_func(*x_reflected)
                print(f'Reflected to {x_expanded} with costs {cost_func(*x_expanded)}')

        # contraction
        else:
            if cost_func(*x_contracted) < cost_func(*x_worst):
                simplex[-1] = x_contracted, cost_func(*x_contracted)
                print(f'Contracted to {x_contracted} with new costs {cost_func(*x_contracted)}')

            # shrink
            else:
                x_shrunk = x_best + (sigma * (simplex['point'][-2:] - x_best)).astype(int)
                simplex['point'][-2:] = x_shrunk
                simplex['cost'][-2:] = [cost_func(*point) for point in simplex['point'][-2:]]
                print(f'Shrunk to:\n{simplex[0]}\n{simplex[1]}\n{simplex[2]}')

    return x_best


# In[7]:


simplex_start = [((75,75), cost_func(75,75)),
                 ((157,8), cost_func(157,8)),
                 ((147,100), cost_func(147,100))]


# random restart, Ecken, günstigen Durchlauf
# Mimimum ist der Lärchenberg

optimum = nelder_mead(data=costs,
                      alpha=1,
                      sigma=0.5,
                      gamma=2,
                      rho=0.5,
                      t=2,
                      points=simplex_start)

print("Optimum:", optimum, '\ncost:', cost_func(*optimum))


# In[8]:


plt.rcParams["figure.figsize"] = (20,10)
plt.contourf(df, levels=255, cmap='terrain_r')
plt.plot(optimum[1], optimum[0], 'rX', markersize=12)
plt.colorbar()
plt.show()


# In[9]:


def function(x):
    a,b = x
    return (1.5 - a + a * b) ** 2 + (2.25 - a + a * b ** 2) ** 2 + (2.625 - a + a * b ** 3) ** 2


# In[10]:


min_wo_gradient = minimize(function, [0, 0], method='CG')
print(min_wo_gradient)
 #     fun: 5.825822747402427e-11
 #     jac: array([-5.26310215e-06, -2.94182541e-06])
 # message: 'Optimization terminated successfully.'
 #    nfev: 76
 #     nit: 10
 #    njev: 19
 #  status: 0
 # success: True
 #       x: array([2.99998094, 0.4999952 ])


# In[11]:


def gradient(x):
    a, b = x
    return 2 * ((1.5 - a + a * b) * np.array([-1 + b, a]) +
                (2.25 - a + a * b ** 2) * np.array([-1 + b ** 2, 2 * a * b]) +
                (2.625 - a + a * b ** 3) * np.array([-1 + b ** 3, 3 * a * b **2]))


# In[12]:


min_with_gradient = minimize(function, [0, 0], method='CG', jac=gradient)
print(min_with_gradient)
 #     fun: 1.6405513553900587e-14
 #     jac: array([ 2.82251570e-07, -1.21100298e-06])
 # message: 'Optimization terminated successfully.'
 #    nfev: 19
 #     nit: 10
 #    njev: 19
 #  status: 0
 # success: True
 #       x: array([2.99999994, 0.49999996])

# In[13]:


# runtime comparison
get_ipython().run_line_magic('timeit', "minimize(function, np.array([0.0, 0.0]), method='CG')")
# 979 µs ± 68 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)


# In[14]:


get_ipython().run_line_magic('timeit', "minimize(function, np.array([0.0, 0.0]), method='CG', jac=gradient)")
# 860 µs ± 30.3 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
