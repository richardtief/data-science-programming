# !/usr/bin/python3

import numpy as np
import time
from math import sqrt
from timeit import default_timer as timer
from multiprocessing import Pool
from euclid import euclid_c


data1 = np.array(np.random.rand(2000, 2000))
data2 = np.array(np.random.rand(2000, 2000))

def euclid_distance_math(x, y):
    return sqrt(sum((i - j) ** 2 for d1, d2 in zip(x, y) for i, j in zip(d1, d2)))

def euclid_distance_vectorized(x, y):
    return sqrt(sum((x.flatten() - y.flatten()) ** 2))

# Splitting numpy arrays into two subsets of same size 
data1_1, data1_2 = data1[:1000, ...], data1[1000:, ...]
data2_1, data2_2 = data1[:1000, ...], data1[1000:, ...]

def main():
    # numpy computation
    start = timer()
    print(f'starting computation with numpy')
    print(np.linalg.norm(data2 - data1))
    end = timer()
    print(f'elapsed time: {end - start}s')
    
    # item per item computation
    start = timer()
    print(f'-------------------------------------------------------------\n'
          f'starting computation item per item')
    print(euclid_distance_math(data1, data2))
    end = timer()
    print(f'elapsed time: {end - start}s')
    
    # parallel computation with two cores
    cpu_cores = 2
    start = timer()
    print(f'-------------------------------------------------------------\n'
          f'starting computation with parallelization on {cpu_cores} cores')
    with Pool(processes=cpu_cores) as pool:
        # using chunksize instead of own splitting at line 20
        res = pool.starmap(euclid_distance_math, [(data1, data2)], chunksize=2)
        print(res[0])   
    end = timer()
    print(f'elapsed time: {end - start}s')
    
    # pushin computation to C
    start = timer()
    print(f'-------------------------------------------------------------\n'
          f'starting computation in C')
    print(euclid_c(data1.tolist(), data2.tolist()))
    end = timer()
    print(f'elapsed time: {end - start}s')
      
    # vectorized computation
    start = timer()
    print(f'-------------------------------------------------------------\n'
          f'starting computation vectorized')
    print(euclid_distance_vectorized(data1, data2))
    end = timer()
    print(f'elapsed time: {end - start}s')
    
if __name__ == '__main__':
    main()
    
"""
starting computation with numpy
816.4517594400052
elapsed time: 0.012906678000000005s
-------------------------------------------------------------
starting computation item per item
816.4517594399805
elapsed time: 2.853860611s
-------------------------------------------------------------
starting computation with parallelization on 2 cores
816.4517594399805
elapsed time: 3.295689055s
-------------------------------------------------------------
starting computation in C
816.4517594399805
elapsed time: 0.8653160710000005s
-------------------------------------------------------------
starting computation vectorized
816.4517594399805
elapsed time: 0.6581074729999994s
"""