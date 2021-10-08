#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# read csv and explore data
df = pd.read_csv("data-cls.csv")
df


# In[3]:


# Calculates the entropy of the given data set for the target attribute
def entropy(dataframe, target):
    return sum([(- (entry / dataframe.shape[0]) * np.log2(entry / dataframe.shape[0])) 
                  for entry in dataframe[target].value_counts()])


# In[4]:


def max_information_gain(dataframe, target):
    current_entropy = entropy(df, target)
    entropies = dict()
    # iterate over dataframe by ignoring target column
    for attribute in dataframe.drop(target, 1):
        # relative frequency
        counts = dataframe[attribute].value_counts(normalize=True)
        # calculate entropies on every element in the series
        calc_entropies = counts.index.to_series().apply(
            lambda i: entropy(dataframe[dataframe[attribute] == i],
                              target))
        entropies[attribute] = current_entropy - ((counts * calc_entropies).sum())
    # return best column with max information gain
    return max(entropies, key=entropies.get)


# In[6]:


def decision_tree(dataframe, target, last_best_attribute=None, indent = ""):
    current_entropy = entropy(dataframe, target)
    if current_entropy == 0:
        indent += "   "
        return f'{indent}-- {dataframe[target].iloc[0]}\n'

    best_attribute = max_information_gain(dataframe, target)
    
    if best_attribute == last_best_attribute:
        indent += "   "
        return f'{indent}-- {dataframe[target].value_counts(normalize=True)}'
    
    tree = f'{indent}WHEN {best_attribute}\n'

    for entry in pd.unique(dataframe[best_attribute]):
        df2 = dataframe[dataframe[best_attribute] == entry]
        
        tree += f'{indent}IS {entry}'
        # recursion
        tree += f'{indent}\n{decision_tree(df2, target, best_attribute, indent="    ")}'
        
    return tree


# In[7]:


print(decision_tree(df, 'tennis'))

