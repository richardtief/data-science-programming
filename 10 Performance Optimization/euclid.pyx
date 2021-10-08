from math import sqrt

def euclid_c(list x,list y):
    return sqrt(sum((i - j) ** 2 for d1, d2 in zip(x, y) for i, j in zip(d1, d2)))
