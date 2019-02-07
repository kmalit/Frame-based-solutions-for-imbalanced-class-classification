from sklearn.datasets import make_classification #Generate random dataset
import numpy as np # Data crunching
from scipy.optimize import nnls # for the frame computations
from tqdm import tqdm #To show computation progress

###############################################################################
# RANDOM DATA GENERATOR
# Generates data with differennt n, dimensions and class weights
###############################################################################
def generateData(n=100,d=5, classWeights = [.80,.20]):
    data = make_classification(n_samples=n,# Nunber of samples
                        n_features=d, # Number of features
                        n_informative=2, # Number of informative features
                        n_redundant=2, # Number of reduntant features
                        n_classes=2, #Number of classes
                        n_clusters_per_class=2, # Number of clusters per class
                        weights=classWeights, # List of weights for the class
                        flip_y=0.01,#raction of samples whose class are randomly exchanged
                        scale=2.5, #Scales the features
                        shuffle=True, random_state=10)
    X,y = data[0],data[1]
    y.shape = (len(data[1]),1)
    return np.concatenate((y,X),axis = 1)
###############################################################################
# Get the Frame of the data
##############################################################################
def frame( X, M=1000.0 ):
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

def frame1( X, M=1000.0 ):
    X = X[:,1:]     
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
def frame2( X, M=1000.0 ):
    X = X[np.where(X[:,0]==0)][:,1:]   
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
def frame3( X, M=1000.0 ):
    X = X[np.where(X[:,0]==1)][:,1:]   
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

###############################################################################

def frameW( X, M=1000.0 ):
    # initialization
    n = X.shape[0]
    q = np.array([],dtype=np.int64)
    Q = np.vstack( ( X.T, M * np.ones(n) ) )
    W = np.zeros((n,n))
    for i in tqdm(range(n)):
        a, rnorm = nnls( Q, Q.T[i] )
        ind = np.where( a > 0 )[0]
        q = np.union1d(q,ind)
        W[i,:] = a
    return q, W

###############################################################################
def kernel_frame( K, M=100.0, tol=1e-8, verbose=False ):
    # initialization
    n = K.shape[0]
    q = np.array([],dtype=np.int64)
    KM = K + M * np.ones_like(K)
    for i in tqdm(range(n)):
        if verbose:
            print('-'*20)
            print('{}/{}'.format(i,n))
        a = fnnls( KM, KM[:,i], tol=tol, verbose=verbose )
        ind = np.where( a > 0 )[0]
        q = np.union1d(q,ind)
    return q

###############################################################################
def fnnls( AtA, Atb, tol=1e-8, verbose=False ):
    n = AtA.shape[0]
    max_it = 3 * n
    x = np.zeros(n)
    s = np.zeros(n)
    P = np.zeros(n,dtype=bool)

    w = Atb - np.dot(AtA,x)
    it = 0

    while np.sum(P) < n and np.any(w[np.invert(P)] > tol) and it < max_it:
        i = np.argmax( w * np.invert(P) )
        if verbose:
            print('choose',i)
        P[i] = True


        s[P] = np.linalg.solve(AtA[np.ix_(P,P)], Atb[P])
        s[ np.invert(P) ] = 0.0

        while np.any( s[P] <= tol ):
            if verbose:
                print('fix')
            it = it + 1

            ind = np.bitwise_and( s <= tol, P )
            a = np.min(x[ind] / (x[ind]-s[ind]))
            x = x + a*(s-x)
            for i in range(n):
                if P[i] and np.abs(x[i]) < tol:
                    P[i] = False

            s[P] = np.linalg.solve(AtA[np.ix_(P,P)], Atb[P])
            s[ np.invert(P) ] = 0.0

        x = np.copy(s)
        w = w = Atb - np.dot(AtA,x)
    if (it == max_it):
        print('Warning: max_it={} reached'.format(max_it))
    return x

###############################################################################