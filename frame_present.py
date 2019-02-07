import numpy as np
from Experiments.generateData import frame
from sklearn.datasets import make_classification #Generate random dataset
from matplotlib import pyplot as plt

###############################################################################
# GENERATE SAMPLE DATA
###############################################################################

df = make_classification(n_samples=100,# Nunber of samples
                        n_features=2, # Number of features
                        n_informative=2, # Number of informative features
                        n_redundant=0, # Number of reduntant features
                        n_classes=2, #Number of classes
                        n_clusters_per_class=1, # Number of clusters per class
                        weights=[0.8,0.2], # List of weights for the class
                        flip_y=0,#raction of samples whose class are randomly exchanged
                        scale=2.5, #Scales the features
                        shuffle=True, random_state=10)
X,y = df[0],df[1]
y.shape = (len(df[1]),1)
df = np.concatenate((y,X),axis = 1)
X,y = df[:,1:],df[:,0]
###############################################################################
# GENERATE FRAMES
###############################################################################
q = frame(X)
fram = X[q]

fram_0 = df[np.where(df[:,0] == 0)][:,1:][frame(df[np.where(df[:,0] == 0)][:,1:])]
fram_1 = df[np.where(df[:,0] == 1)][:,1:][frame(df[np.where(df[:,0] == 1)][:,1:])]


###############################################################################
# PLOTS 
###############################################################################

# Imbalanced class plot
color= ['red' if l == 0 else 'blue' for l in y]
plt.scatter(X[:,0],X[:,1], color=color)
plt.title("Unbalanced class classification", loc = 'center')
plt.savefig('imbalanced_class.png')

# PLot FurthestSUM
FS = X[chosen]
plt.scatter(X[:,0],X[:,1], color='b', alpha = 0.5)
plt.scatter(FS[:,0],FS[:,1],color='r')
plt.title("The Frame", loc = 'center')
#plt.savefig('the_frame.png')


# Frame plot
plt.scatter(X[:,0],X[:,1], color='b', alpha = 0.5)
plt.scatter(fram[:,0],fram[:,1],color='r')
plt.title("The Frame", loc = 'center')
plt.savefig('the_frame.png')

# Frame for individual classes:
color1= ['red' if l == 0 else 'blue' for l in y]
plt.scatter(X[:,0],X[:,1], color=color1, alpha = 0.4)
#plt.scatter(fram[:,0],fram[:,1],color='g')
plt.scatter(fram_0[:,0],fram_0[:,1],color='g')
plt.scatter(fram_1[:,0],fram_1[:,1],color='purple')
plt.title("Frame for each class", loc = 'center')
plt.savefig('each_class_frame.png')
