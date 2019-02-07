import pickle
import numpy as np
import matplotlib as mpl
mpl.use ('Agg')
import matplotlib.pylab as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
###############################################################################
# config datapath
###############################################################################
def get_datapath():
    return "./data/"
###############################################################################
# save and load experiments
###############################################################################
def save_experiment(results, name):
    file_name = get_datapath() + "experiments/" + name + ".pkl"
    with open(file_name, "wb") as file:
        pickle.dump(results, file)

def load_experiment(name):
    file_name = get_datapath() + "experiments/" + name + ".pkl"
    with open(file_name, "rb") as file:
        results = pickle.load(file)
    return results
###############################################################################
# Furthest Sum
###############################################################################
def frthst_sum_f_frame(X,q,size):
    n = X.shape[0]
    idx = []
    if size < len(q):
        d = np.linalg.norm( X[q][0] - X[q], axis=1 )
        l = d.argmax()
        d = np.linalg.norm( X[q][l] - X[q], axis=1 ) 
        i = np.argwhere(X==X[q][d.argmax()])[0][0] 
        idx.append(i)
        pool = list(q)
        pool.remove(i)
        while len(idx) < size:
            d = []
            for j in pool:
                # compute sum of distances to all chosen points
                d.append( np.linalg.norm( X[idx] - X[j], axis=1 ).sum() )
            # pick index of furthest point
            i = pool[ np.array(d).argmax() ]
            pool.remove(i)
            idx.append(i)
    elif size == len(q):
        idx = q
    else:
        idx = list(q)
        pool = np.setdiff1d(list(range(n)),idx)
        while len(idx) < size:
            d = []
            for j in pool:
                # compute sum of distances to all chosen points
                d.append( np.linalg.norm( X[idx] - X[j], axis=1 ).sum() )
                # pick index of furthest point
            i = pool[ np.array(d).argmax() ]
            idx.append(i)
            pool = np.setdiff1d(list(range(n)),idx)
    return idx

###############################################################################
# Train test split
###############################################################################
def t_t_split(X,p = 0.8):
    train_ind = np.random.choice(len(X),int(p*len(X)),replace=False)
    return train_ind

###############################################################################
# Function to perform a grid search and return best model score
###############################################################################

def Gridsearch(model,param_grid,data):
    X,y = data[:,1:],data[:,0]
    grid = GridSearchCV(model,param_grid, cv=5, refit=True, verbose=False,iid = False)
    grid.fit(X,y)
    bst_params = grid.best_params_
    return bst_params
    
###############################################################################
# Function to get detailed classification report
###############################################################################
def classreport(data,model,title):
    X,y = data[:,1:],data[:,0]
    report_data = []
    report = classification_report(y, model.predict(X))
    lines = report.split('\n')
    for line in lines[2:4]:
        row = {}
        row_data = line.split('      ')
        row['model'] = title 
        row['class'] = row_data[1]
        row['precision'] = float(row_data[2])
        row['recall'] = float(row_data[3])
        row['support'] = float(row_data[5])
        report_data.append(row)
    return report_data

###############################################################################
# Function to print the sores from the classifiers
###############################################################################
def get_auc_scores(data, model):
    X,y = data[:,1:],data[:,0]
    auc_score = roc_auc_score(y, model.predict(X)); 
    fpr_df, tpr_df, _ = roc_curve(y, model.predict(X)); 
    return (auc_score, fpr_df, tpr_df)

###############################################################################
# Plot AUC Scores
###############################################################################
def plot_auc(FPR,TPR,AUC,PLOTNAMES):
    plt.figure(figsize = (12,6), linewidth= 1)
    plt.plot([0,1], [0,1], 'k--', label = 'Random: 0.5')
    for i in range (len(FPR)):
        plt.plot(FPR[i], TPR[i], label = f"'{PLOTNAMES[i]}'" + str(round(AUC[i],5)))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(PLOTNAMES+' ROC Curve')
    plt.legend(loc='best')
    plt.savefig(PLOTNAMES+'.png')
