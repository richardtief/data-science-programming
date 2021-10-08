#!/usr/bin/env python
# coding: utf-8

# In[1]:


item_dict = {
    1: "5-Minute-Noodles",
    2: "Pineapple",
    3: "Applesauce",
    4: "Asia-Snack",
    5: "Cup",
    6: "Beer",
    7: "College pad",
    8: "Canned tomatoes",
    9: "Peas",
    10: "Peas & Carrots",
    11: "Lint roller",
    12: "Putty", 
    13: "Mushroom-Spaghetti",
    14: "Salt sticks",
    15: "Mustard",
    16: "Spaghetti",
    17: "Cream",
    18: "Deodorant",
    19: "Disinfection", # !
    20: "Showergel",
    21: "Shot",
    22: "Gummi bears",
    23: "Hair tie",
    24: "Hazelnut waffle",
    25: "Glue stick",
    26: "Cap bomb",
    27: "Air nozzle",
    28: "Sticky note",
    29: "Maoam",
    30: "Milk bar",
    31: "Mouth wash",
    32: "Shaver",
    33: "Small bowl",
    34: "Chocolate bar",
    35: "Pen",
    36: "Tomato paste",
    37: "Coaster",
    38: "Toothpaste",
    39: "Envelopes",
    40: "Deco balls",
    41: "Scented candles",
    42: "Napkins",
    43: "Cereal",
    44: "Cactus",
    45: "Bend lights",
    46: "Booklet",
    47: "OREO cookies",
    48: "Light bag",
    49: "Puzzle",
    50: "Novels",
    51: "Cards",
    52: "Detergent",
    53: "Tissues"
} 


# In[2]:


import pandas as pd
import numpy as np
from itertools import combinations


# In[3]:


items = pd.read_table('items.tsv', sep=" ", header=None)
items = items.to_numpy()

# create dict including transactions
data1 = dict()
for i, t in enumerate(items):
    data1[i+1] = t[~np.isnan(t)].astype(int).tolist()             
              
# explore transactions
for k, transaction in data1.items():
    print(k, ":", transaction)



# Out[3]:


# 1 : [4, 6, 7, 8, 16, 17, 18, 19, 20, 21, 22, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 43, 44, 45, 47, 48]
# 2 : [4, 6, 7, 8, 12, 16, 18, 19, 20, 21, 22, 24, 30, 32, 34, 35, 41, 47, 49, 50, 53]
# 3 : [6, 8, 11, 12, 19, 20, 21, 22, 27, 29, 30, 32, 33, 34, 37, 40, 41, 42, 46, 50, 53]
# 4 : [4, 6, 8, 9, 20, 21, 22, 24, 29, 30, 33, 34, 43, 45, 47, 51, 53]
# 5 : [2, 4, 6, 15, 21, 22, 24, 29, 30, 31, 34, 37, 40, 41, 44, 51, 52, 53]
# 6 : [1, 6, 13, 22, 24, 27, 29, 32, 34, 37, 41, 43, 45, 53]
# 7 : [4, 5, 6, 13, 20, 22, 23, 24, 27, 30, 31, 34, 37, 42, 46, 48, 51]
# 8 : [1, 6, 14, 15, 22, 24, 25, 27, 28, 29, 30, 33, 35, 36, 37, 41, 43, 47, 51]
# 9 : [3, 4, 5, 6, 11, 21, 22, 23, 24, 25, 26, 27, 28, 29, 34, 40, 43, 44, 45, 48, 50, 51]
# 10 : [6, 10, 14, 15, 18, 20, 21, 26, 27, 28, 30, 32, 33, 34, 35, 36, 37, 38, 39, 41, 43, 49, 50, 53]



# In[4]:


# transactions have diffrent length
# skip empty "bad" lines when read file into a dataframe
retail = pd.read_csv('retail.tsv', sep=" ", error_bad_lines=False, warn_bad_lines=False)
retail = retail.to_numpy()

# create dict including transactions
data2 = dict()
for i, t in enumerate(retail):
    data2[i] = t[~np.isnan(t)].astype(int).tolist()
    
# explore transactions
for k, transaction in data2.items():
    print(k, ":", transaction)


# Out[4]:


# 0 : [30, 31, 32]
# 1 : [33, 34, 35]
# 2 : [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
# 3 : [38, 39, 47, 48]
# 4 : [38, 39, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]
# 5 : [32, 41, 59, 60, 61, 62]
# 6 : [3, 39, 48]
# 7 : [63, 64, 65, 66, 67, 68]
# 8 : [32, 69]
# 9 : [48, 70, 71, 72]
# 10 : [39, 73, 74, 75, 76, 77, 78, 79]
# 11 : [36, 38, 39, 41, 48, 79, 80, 81]
# 12 : [82, 83, 84]
# 13 : [41, 85, 86, 87, 88]
# 14 : [39, 48, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101]
# 15 : [36, 38, 39, 48, 89]
# 16 : [39, 41, 102, 103, 104, 105, 106, 107, 108]
# 17 : [38, 39, 41, 109, 110]
# 18 : [39, 111, 112, 113, 114, 115, 116, 117, 118]
# .
# .
# .


# In[26]:


def APRIORI(data, min_support):
    candidates = {(item,) for transaction in data.values() for item in transaction}
    sorted_candidates = sorted(candidates)
    
    k = 1
    result = dict() # key: length freq item set, value: freq item sets
    
    while sorted_candidates:
        counts = dict()
        
        for i, transaction in data.items():
            for candidate_in_transaction in combinations(transaction, r=k):
                
                if k == 1 or candidate_in_transaction in sorted_candidates:
                    
                    if candidate_in_transaction not in counts:
                        counts[candidate_in_transaction] = 0
                    # increase candidate count
                    counts[candidate_in_transaction] += 1
          
        frequent_items = set()
        
        for item, count in sorted(counts.items()):
        	# minimum support fulfilled --> add to set
            if count / len(data) >= min_support:
                frequent_items.add(item)

        result[k] = (sorted(frequent_items))

        relevant_items = sorted({item for items in frequent_items for item in items})
        # increase length of freq item set
        k += 1
        sorted_candidates = list(combinations(relevant_items, r=k))
    
    return result


# In[6]:


for k, fi in APRIORI(data1, min_support=0.7).items():
    freq = list()
    for i in fi:
        f = [item_dict[item] for item in i]
        freq.append(tuple(f))
    print('\nk =', k, ":", str(len(freq)), "\n" + str(fi), "\n"  + str(freq), "\n")


# Out[6]:


# k = 1 : 8 
# [(6,), (21,), (22,), (24,), (30,), (34,), (37,), (41,)] 
# [('Beer',), ('Shot',), ('Gummi bears',), ('Hazelnut waffle',), ('Milk bar',), ('Chocolate bar',), ('Coaster',), ('Scented candles',)] 


# k = 2 : 12 
# [(6, 21), (6, 22), (6, 24), (6, 30), (6, 34), (6, 37), (6, 41), (21, 34), (22, 24), (22, 30), (22, 34), (30, 34)] 
# [('Beer', 'Shot'), ('Beer', 'Gummi bears'), ('Beer', 'Hazelnut waffle'), ('Beer', 'Milk bar'), ('Beer', 'Chocolate bar'), ('Beer', 'Coaster'), ('Beer', 'Scented candles'), ('Shot', 'Chocolate bar'), ('Gummi bears', 'Hazelnut waffle'), ('Gummi bears', 'Milk bar'), ('Gummi bears', 'Chocolate bar'), ('Milk bar', 'Chocolate bar')] 


# k = 3 : 5 
# [(6, 21, 34), (6, 22, 24), (6, 22, 30), (6, 22, 34), (6, 30, 34)] 
# [('Beer', 'Shot', 'Chocolate bar'), ('Beer', 'Gummi bears', 'Hazelnut waffle'), ('Beer', 'Gummi bears', 'Milk bar'), ('Beer', 'Gummi bears', 'Chocolate bar'), ('Beer', 'Milk bar', 'Chocolate bar')] 



# In[7]:


for k, fi in APRIORI(data2, min_support=0.1).items():
    freq = list()
    for i in fi:
        f = [item_dict[item] for item in i]
        freq.append(tuple(f))
    print('\nk =', k, ":", str(len(freq)), "\n" + str(fi), "\n"  + str(freq), "\n")


# Out[7]:


# k = 1 : 5 
# [(32,), (38,), (39,), (41,), (48,)] 
# [('Shaver',), ('Toothpaste',), ('Envelopes',), ('Scented candles',), ('Light bag',)] 


# k = 2 : 3 
# [(38, 39), (39, 41), (39, 48)] 
# [('Toothpaste', 'Envelopes'), ('Envelopes', 'Scented candles'), ('Envelopes', 'Light bag')] 


# k = 3 : 0 
# [] 
# []


# In[8]:


def ECLAT(data, min_support):
    k = 1
    item_transactions = dict()
    
    for i, transaction in data.items():

        for item in transaction:
            if (item,) not in item_transactions:
                item_transactions[item,] = set()
            item_transactions[item,].add(i)
            
    result = dict()
    
    while item_transactions:
        k +=1
        
        frequent_items = set()
        
        for item, transaction_indices in  item_transactions.items():
            if len(transaction_indices) / len(data) >= min_support:
                frequent_items.add(item)
        result[k] = (sorted(frequent_items))
        
        relevant_items = sorted({item for items in frequent_items for item in items})
        candidates = list(combinations(relevant_items, r=k))
        new_item_transactions = dict()
        
        for candidate in candidates:
            old_candidates = list(combinations(candidate, r=k-1))
            new_item_transactions[candidate] = item_transactions[old_candidates[0]].intersection(*(item_transactions[i] for i in old_candidates[1:]))
        
        item_transactions = new_item_transactions
        
    return result


# In[9]:


for k, fi in ECLAT(data1, min_support=0.7).items():
    freq = list()
    for i in fi:
        f = [item_dict[item] for item in i]
        freq.append(tuple(f))
    print('\nk =', k, ":", str(len(freq)), "\n" + str(fi), "\n"  + str(freq), "\n")

# Out[9]:


# k = 2 : 8 
# [(6,), (21,), (22,), (24,), (30,), (34,), (37,), (41,)] 
# [['Beer'], ['Shot'], ['Gummi bears'], ['Hazelnut waffle'], ['Milk bar'], ['Chocolate bar'], ['Coaster'], ['Scented candles']] 


# k = 3 : 12 
# [(6, 21), (6, 22), (6, 24), (6, 30), (6, 34), (6, 37), (6, 41), (21, 34), (22, 24), (22, 30), (22, 34), (30, 34)] 
# [['Beer', 'Shot'], ['Beer', 'Gummi bears'], ['Beer', 'Hazelnut waffle'], ['Beer', 'Milk bar'], ['Beer', 'Chocolate bar'], ['Beer', 'Coaster'], ['Beer', 'Scented candles'], ['Shot', 'Chocolate bar'], ['Gummi bears', 'Hazelnut waffle'], ['Gummi bears', 'Milk bar'], ['Gummi bears', 'Chocolate bar'], ['Milk bar', 'Chocolate bar']] 


# k = 4 : 5 
# [(6, 21, 34), (6, 22, 24), (6, 22, 30), (6, 22, 34), (6, 30, 34)] 
# [['Beer', 'Shot', 'Chocolate bar'], ['Beer', 'Gummi bears', 'Hazelnut waffle'], ['Beer', 'Gummi bears', 'Milk bar'], ['Beer', 'Gummi bears', 'Chocolate bar'], ['Beer', 'Milk bar', 'Chocolate bar']] 


# k = 5 : 0 
# [] 
# [] 



# In[10]:


for k, fi in ECLAT(data2, min_support=0.1).items():
    freq = list()
    for i in fi:
        f = [item_dict[item] for item in i]
        freq.append(tuple(f))
    print('\nk =', k, ":", str(len(freq)), "\n" + str(fi), "\n"  + str(freq), "\n")

# Out[10]:


# k = 2 : 5 
# [(32,), (38,), (39,), (41,), (48,)] 
# [['Shaver'], ['Toothpaste'], ['Envelopes'], ['Scented candles'], ['Light bag']] 


# k = 3 : 3 
# [(38, 39), (39, 41), (39, 48)] 
# [['Toothpaste', 'Envelopes'], ['Envelopes', 'Scented candles'], ['Envelopes', 'Light bag']] 


# k = 4 : 0 
# [] 
# [] 


# In[11]:


print("APRIORI(data1, min_support=0.7):")
get_ipython().run_line_magic('timeit', 'APRIORI(data1, min_support=0.7)')
print("\nAPRIORI(data2, min_support=0.1):")
get_ipython().run_line_magic('timeit', 'APRIORI(data2, min_support=0.1)')
print("\nECLAT(data1, min_support=0.7):")
get_ipython().run_line_magic('timeit', 'ECLAT(data1, min_support=0.7)')
print("\nECLAT(data2, min_support=0.1):")
get_ipython().run_line_magic('timeit', 'ECLAT(data2, min_support=0.1)')


# Out[11]:


# APRIORI(data1, min_support=0.7):
# 42 ms ± 2.73 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

# APRIORI(data2, min_support=0.1):
# 6.82 s ± 366 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# ECLAT(data1, min_support=0.7):
# 241 µs ± 2.04 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

# ECLAT(data2, min_support=0.1):
# 382 ms ± 16.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

