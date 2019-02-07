#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 13:34:27 2019

@author: keldine
"""
##############################################################################
#Libaries import
##############################################################################
import numpy as np
##############################################################################
# START ENGINES
##############################################################################
import ipyparallel as ipp
print ("Starting Engines")
rc = ipp.Client()
rc.ids
dview = rc[:]
print (len(rc.ids),"Engines Started")
###############################################################################
# Library imports
###############################################################################
print ("Loading libraries into engines")
with dview.sync_imports():
    import numpy # Data crunching
    from scipy.optimize import nnls # for the frame computations
    from tqdm import tqdm #
print ("Libraries successfully loaded into engines")

###############################################################################
# Partitioning function
###############################################################################
def partition(lst, n):
    division = len(lst) / n
    return [lst[round(division * i):round(division * (i + 1))] for i in range(n)]
###############################################################################
# Function to get indices from each partition
###############################################################################
def frame_indices(Q,Range):
    q = numpy.array([],dtype=numpy.int64)
    for i in Range:
        a, rnorm = nnls( Q, Q.T[i] )
        ind = numpy.where( a > 0 )[0]
        q = numpy.union1d(q,ind)
    return q

###############################################################################
# function to compute indices and merge them
###############################################################################
def frame_multicore(X, M=1000.0):
    np.random.seed(10)
    # initialization
    X = X[:,1:]
    eng_count = len(rc.ids)
    n = X.shape[0]
    q = np.array([],dtype=numpy.int64)
    Q = np.vstack( ( X.T, M * np.ones(n) ) )
    #Split the data into n where n is the number of available engines
    Range = partition(range(n),eng_count)
    list_Q = [Q]*eng_count
    #Map the indices worl    
    ind = dview.map(frame_indices,list_Q,Range)
    ind = ind.get()
    # Merge the results
    ind = [y for x in ind for y in x]
    q = np.union1d(q,ind)
    return q

###############################################################################
# Single core index computation function
###############################################################################
def frame(X, M=1000.0):
    np.random.seed(10)
    # initialization
    n = X.shape[0]
    q = np.array([],dtype=np.int64)
    Q = np.vstack( ( X.T, M * np.ones(n) ) )
    
    for i in tqdm(range(n)):
        a, rnorm = nnls( Q, Q.T[i] )
        ind = np.where( a > 0 )[0]
        q = np.union1d(q,ind)
    return q
