#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from timeit import timeit


# In[2]:


df = pd.read_csv("data-OLS.csv")
df


# In[3]:


plt.rcParams["figure.figsize"] = (10,5)
df.plot.scatter(x='x', y='y', color="grey")
plt.show()


# In[4]:


# OLS function for OLS using scipy.optimize.minimize

def SSE(params):
    a1, a0 = params
    x = df['x']
    y = df['y']
    return ((y - (a1 * x + a0)) ** 2).sum()

def optimize_minimize():
    return optimize.minimize(SSE, np.array(([0,1])))['x']


# In[5]:


# apply optimize.minimize function

a1, a0 = optimize_minimize()
print("a1:", a1)
print("a0:", a0)


# In[6]:


# plot with optimize.minimize parameter
plt.rcParams["figure.figsize"] = (10,5)

x = np.linspace(0, 1, 2)
df.plot.scatter(x="x", y="y", color="grey")
plt.plot(x, a1 * x + a0, color="darkred")
plt.show()


# In[7]:


# OLS function for OLS using matrices

def matrix_solution():
    x_transposed = np.array((df['x'].to_numpy(), 
                             np.ones_like(df['x'].to_numpy())))
    y_transposed = np.array((df['y'].to_numpy(), 
                             np.ones_like(df['y'].to_numpy())))
    # Compute the (multiplicative) inverse of a matrix.
    # @ is the matrix multiplicator
    A = np.linalg.inv(x_transposed @ x_transposed.T) @ x_transposed @ y_transposed.T
    return A[0]


# In[8]:


a1, a0 = matrix_solution()
print("a1:", a1)
print("a0:", a0)


# In[9]:


# plot with matrix solution parameter
plt.rcParams["figure.figsize"] = (10,5)

x = np.linspace(0, 1, 2)
df.plot.scatter(x="x", y="y", color="grey")
plt.plot(x, a1 * x + a0, color="darkred")
plt.show()


# In[10]:


# compare runtime between minimize and matrix solution
get_ipython().run_line_magic('timeit', 'optimize_minimize()')
get_ipython().run_line_magic('timeit', 'matrix_solution()')


# In[11]:


# set lambda
lambda_1 = 2
lambda_2 = 4


# ridge regression
def SSE_ridge_regression(params):
    a1, a0 = params
    return SSE(params) + lambda_1 * (a1 ** 2 +a0 ** 2)
def optimize_minimize_ridge():
    return optimize.minimize(SSE_ridge_regression, np.array(([0,1])))['x']


# lasso regression
def SSE_lasso_regression(params):
    a1, a0 = params
    return SSE(params) + lambda_2 * (abs(a1) + abs(a0))
def optimize_minimize_lasso():
    return optimize.minimize(SSE_lasso_regression, np.array(([0,1])))['x'] 


# elastic net 
def SSE_elastic_net(params):
    a1, a0 = params
    return SSE(params) + lambda_1 * (abs(a1) + abs(a0)) + lambda_2 * (a1 ** 2 + a0 ** 2)
def optimize_minimize_elastic_net():
    return optimize.minimize(SSE_elastic_net, np.array(([0,1])))['x'] 


# In[12]:


# plot to compare optimize and regularization methods
plt.rcParams["figure.figsize"] = (10,5)

x = np.linspace(0, 1, 2)
df.plot.scatter(x="x", y="y", color="grey")


a1, a0 = optimize_minimize()
plt.plot(x, a1 * x + a0, label="OLS minimize")
a1, a0 = matrix_solution()
plt.plot(x, a1 * x + a0, label = "OLS matrix")
a1, a0 = optimize_minimize_ridge()    
plt.plot(x, a1 * x + a0, label= "OLS ridge regression")
a1, a0 = optimize_minimize_lasso()    
plt.plot(x, a1 * x + a0, label= "OLS lasso regression")
a1, a0 = optimize_minimize_elastic_net()    
plt.plot(x, a1 * x + a0, label= "OLS elastic net")

plt.legend()
plt.show()


# In[13]:


# compare runtime among all solutions
print("OLS minimize:")
get_ipython().run_line_magic('timeit', 'optimize_minimize()')
print("\nOLS matrix:")
get_ipython().run_line_magic('timeit', 'matrix_solution()')
print("\nOLS ridge regression:")
get_ipython().run_line_magic('timeit', 'optimize_minimize_ridge()')
print("\nOLS lasso regression:")
get_ipython().run_line_magic('timeit', 'optimize_minimize_lasso()')
print("\nOLS elastic net:")
get_ipython().run_line_magic('timeit', 'optimize_minimize_elastic_net()')

