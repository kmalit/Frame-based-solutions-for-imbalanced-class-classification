# Set Up System Paths
import sys
import os
module_path = os.path.abspath(os.getcwd())
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)
# Data wranglers
import numpy as np
# Parallel computing
import ipyparallel as ipp
# Helper functions
from Experiments.helperfuncs import save_experiment,load_experiment

###############################################################################
# Start engines for parallel computation
###############################################################################
rc = ipp.Client()
dview = rc[:]

###############################################################################
# Load the generated datasets, compute the frames and save the frame indices
###############################################################################

# Load libraries into engines
print ("Loading frame computation libraries into engines")
with dview.sync_imports():
    import numpy # Data crunching
    from scipy.optimize import nnls # for the frame computation
print ("Libraries successfully loaded into engines")
   
# Partitioning function
def partition(lst, n):
    division = len(lst) / n
    return [lst[round(division * i):round(division * (i + 1))] for i in range(n)]
   
# Function to get indices from each partition
def frame_indices(Q,Range):
    q = numpy.array([],dtype=numpy.int64)
    for i in Range:
        a, rnorm = nnls( Q, Q.T[i] )
        ind = numpy.where( a > 0 )[0]
        q = numpy.union1d(q,ind)
    return q

# Function to generate frames
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

# Lad generated datasets
DATASETS = load_experiment('DATASETS')
# Compute frame for entire dataset
print("entering calculation of frame for full datasets")
q = list(map(frame_multicore,DATASETS))
save_experiment(q,"full_frames")
print("exiting calculation of frame for full datasets")