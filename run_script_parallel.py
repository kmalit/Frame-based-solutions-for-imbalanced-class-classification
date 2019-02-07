# Set Up System Paths
import sys
import os
module_path = os.path.abspath(os.getcwd())
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)

# Parallel computing
import ipyparallel as ipp
###############################################################################
# Start engines for parallel computation
###############################################################################
print("Starting Engines")
rc = ipp.Client()
dview = rc[:]
print(len(rc.ids)," Engines Started")
###############################################################################

print ("Loading libraries and helper functions into engines")
with dview.sync_imports():
    # Data wranglers
    import pandas as pd
    import numpy as np

    # Fit models
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.svm import SVC

    # Helper functions
    import operator
    from functools import reduce
    from Experiments.generateData import generateData
    from Experiments.generateData import frame1,frame2,frame3
    from Experiments.helperfuncs import save_experiment,load_experiment
    from Experiments.helperfuncs import t_t_split
    from Experiments.helperfuncs import frthst_sum_f_frame
    from Experiments.helperfuncs import Gridsearch
    from Experiments.helperfuncs import classreport
print ("Libraries and helper functions successfully loaded into engines")
###############################################################################
# Define Variables
###############################################################################

print("Defining variables...")
# DATASETS TO BE GENERATED
DATASET_NAMES = ["5k_95_5_6d","5k_85_15_6d","5k_70_30_6d","10k_95_5_7d","10k_85_15_7d","10k_70_30_7d",
                 "15k_95_5_9d","15k_85_15_9d","15k_70_30_9d"]

#DATASET PARAMETERS
n = [5000,5000,5000,10000,10000,10000,15000,15000,15000]
#n = [300,300,300,400,400,400,500,500,500]
d = [6,6,6,7,7,7,9,9,9]
w = [[.95,.05],[.85,.15],[.70,.30],[.95,.05],[.85,.15],[.70,.30],[.95,.05],[.85,.15],[.70,.30]]

# MODELS TO BE FITTED
MODELS = [SVC(),RandomForestClassifier(),XGBClassifier()]
TRAINED_MODELS = [[SVC(),SVC(),SVC(),SVC(),SVC(),SVC(),SVC(),SVC(),SVC()],[RandomForestClassifier(),
                   RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier(),
                   RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier(),
                   RandomForestClassifier()],[XGBClassifier(),XGBClassifier(),XGBClassifier(),XGBClassifier(),
                                         XGBClassifier(),XGBClassifier(),XGBClassifier(),XGBClassifier(),XGBClassifier()]]
# MODEL NAMES
MODEL_NAME = ['SVM','RF','GB_DT']

# MODEL PARAMETERS 
M_PARAMS = [{'kernel':['linear'],'gamma': [0.1, 0.01, 0.001,1,1.5,5,10]},
              {'max_depth': [3, 5, 6, 7],'min_samples_split': [3, 5, 6, 7],'n_estimators':[10,50,100]},
              {'max_depth': [5,6,7,8], 'gamma': [0.1, 0.01, 0.001],'learning_rate': [0.05,0.1, 0.2, 0.3]}]
print("Variables successfully defined")
###############################################################################
# Generate and save the various datasets
###############################################################################
print("Defining functions..")
def generate_data():
    print("generating datasets...")
    DATASETS = dview.map_sync(generateData,n,d,w)
    save_experiment(DATASETS,'DATASETS')
    print("Completed dataset generation")
    return None

###############################################################################
# Load the generated datasets, compute the frames and save the frame indices
###############################################################################

def calc_frame():    
    DATASETS = load_experiment('DATASETS')
    # Compute frame for entire dataset
    print("entering calculation of frame for full datasets")
    q = dview.map_sync(frame1,DATASETS)
    save_experiment(q,"full_frames")
    print("exiting calculation of frame for full datasets")
    
    # Compute Frame for majority class
    print("entering calculation of frame for majority class")
    q_0 = dview.map_sync(frame2,DATASETS)
    save_experiment(q_0, "majority_class_frames")
    print("exiting calculation of frame for majority class")
    
    # Compute frame for minority class
    print("entering calculation of frame for minority class")
    q_1 = dview.map_sync(frame3,DATASETS)
    save_experiment(q_1, "minority_class_frames")
    print("exiting calculation of frame for minority class")
    return None

###############################################################################
# Train test split
###############################################################################
def train_test_split ():
    #Split full datasets
    DATASETS = load_experiment('DATASETS')
    print("Train test split for full dataset...")
    train_ind = dview.map_sync(t_t_split,DATASETS)
    train_full_df = dview.map_sync(lambda x,ind:x[ind],DATASETS,train_ind)
    test_ind = dview.map_sync(lambda x1,x2: list(set(list(range(len(x1))))-set(x2)), DATASETS,train_ind)
    test_full_df  = dview.map_sync(lambda x,ind: x[ind],DATASETS,test_ind)
    save_experiment(train_full_df,"train_full_df")
    save_experiment(test_full_df,"test_full_df")
    print("Train test split for full dataset completed")
    
    #Split for frame on full dataset based on furthest sum approach
    print("Train test split for full frame...")
    full_frame_indices = load_experiment('full_frames') #confirm version
    train_ind = dview.map_sync(frthst_sum_f_frame,DATASETS,full_frame_indices,list(map(int,np.multiply(n,0.8))))
    train_df = dview.map_sync(lambda x,ind:x[ind],DATASETS,train_ind)
    test_ind = dview.map_sync(lambda x1,x2: list(set(list(range(len(x1))))-set(x2)), DATASETS,train_ind)
    test_df  = dview.map_sync(lambda x,ind: x[ind],DATASETS,test_ind)
    save_experiment(train_df,"train_from_fframe")
    save_experiment(test_df,"test_from_fframe")
    print("Train test split for full frame completed")
    
    #Split from frame of each class 
    print("Train test split for separate frames by class...")
    majCls_f_ind = load_experiment('majority_class_frames')
    minCls_f_ind = load_experiment('minority_class_frames')
    ind1 = dview.map_sync(frthst_sum_f_frame,DATASETS,majCls_f_ind,list(map(int,np.multiply(np.multiply([i[0] for i in w],0.8),n))))

    ind2 = dview.map_sync(frthst_sum_f_frame,DATASETS,minCls_f_ind,list(map(int,np.multiply(np.multiply([i[1] for i in w],0.8),n))))

    train_ind = dview.map_sync(lambda x1,x2: x1+x2, ind1,ind2)
    train_df = train_df = dview.map_sync(lambda x,ind:x[ind],DATASETS,train_ind)
    test_ind = dview.map_sync(lambda x1,x2: list(set(list(range(len(x1))))-set(x2)), DATASETS,train_ind)
    test_df  = dview.map_sync(lambda x,ind: x[ind],DATASETS,test_ind)    
    save_experiment(train_df,"train_from_CLframe")
    save_experiment(test_df,"test_from_CLframe")   
    print("Train test split for separate frames completed")
    return None

###############################################################################
# Carry out grid search to get best model for each dataset and fit best model
###############################################################################
def fit_best_m():
    train_full_df = load_experiment('train_full_df')
    train_from_fframe = load_experiment('train_from_fframe')
    train_from_CLframe = load_experiment('train_from_CLframe')
    i = 0
    for model in MODELS:
        print(f"entering grid search for the '{MODEL_NAME[i]}'")
        # Grid search and model fit for full data
        best_prams_fdf = list(map(Gridsearch,[model]*len(DATASET_NAMES),[M_PARAMS[i]]*len(DATASET_NAMES),train_full_df))
        bst_mod_fdf = []
        for j in range(len(train_full_df)):
            fitted = TRAINED_MODELS[i][j].set_params(**best_prams_fdf[j])
            bst_mod_fdf.append(fitted)
        models = []
        for j in range(len(train_full_df)):
            fitted = bst_mod_fdf[j].fit(train_full_df[j][:,1:],train_full_df[j][:,0])
            models.append(fitted)
        save_experiment(models, MODEL_NAME[i] + '_' + "best_model_f_full_df")
        
        # Grid search and model fit for frame data
        best_prams_ffdf = list(map(Gridsearch,[model]*len(DATASET_NAMES),[M_PARAMS[i]]*len(DATASET_NAMES),train_from_fframe))
        bst_mod_ffdf = []
        for j in range(len(train_from_fframe)):
            fitted = TRAINED_MODELS[i][j].set_params(**best_prams_ffdf[j])
            bst_mod_ffdf.append(fitted)
        models1 = []
        for j in range(len(train_from_fframe)):
            fitted = bst_mod_ffdf[j].fit(train_from_fframe[j][:,1:],train_from_fframe[j][:,0])
            models1.append(fitted)
        save_experiment(models1, MODEL_NAME[i] + '_' + "best_model_f_full_frame")
        
        # Grid search and model fit for class frame data
        best_prams_CLfdf = list(map(Gridsearch,[model]*len(DATASET_NAMES),[M_PARAMS[i]]*len(DATASET_NAMES),train_from_CLframe))
        bst_mod_fdf = []
        for j in range(len(train_from_CLframe)):
            fitted = TRAINED_MODELS[i][j].set_params(**best_prams_CLfdf[j])
            bst_mod_fdf.append(fitted)
        models2 = []
        for j in range(len(train_from_CLframe)):
            fitted = bst_mod_fdf[j].fit(train_from_CLframe[j][:,1:],train_from_CLframe[j][:,0])
            models2.append(fitted)
        save_experiment(models2, MODEL_NAME[i] + '_' + "best_model_f_Class_frames")
        print(f"exiting grid search for the '{MODEL_NAME[i]}'")
        i +=1
    return None
###############################################################################
# Check fit scores for each model for each dataset and save as csv
###############################################################################
def check_train_fit():
    train_full_df = load_experiment('train_full_df')
    train_from_fframe = load_experiment('train_from_fframe')
    train_from_CLframe = load_experiment('train_from_CLframe')
    DF = []
    i = 0
    for model in MODELS:
        print(f"checking train fit for the '{MODEL_NAME[i]}' model")
        # train fit for full data models
        mod1 = load_experiment(MODEL_NAME[i] + '_' + "best_model_f_full_df")
        train_res1 = list(map(classreport,train_full_df,mod1,[f"{MODEL_NAME[i]}_df_" + s for s in DATASET_NAMES]))
        train_res1 = reduce(operator.concat, train_res1)
        # train fit for full frame models 
        mod2 = load_experiment(MODEL_NAME[i] + '_' + "best_model_f_full_frame")
        train_res2 = list(map(classreport,train_from_fframe,mod2,[f"{MODEL_NAME[i]}_ff_" + s for s in DATASET_NAMES]))
        train_res2 = reduce(operator.concat, train_res2)
        # train fit for by class frame models 
        mod3 = load_experiment(MODEL_NAME[i] + '_' + "best_model_f_Class_frames")
        train_res3 = list(map(classreport,train_from_CLframe,mod3,[f"{MODEL_NAME[i]}_fCf_" + s for s in DATASET_NAMES]))
        train_res3 = reduce(operator.concat, train_res3)
        
        # Combine all together
        DF = DF + train_res1+train_res2+train_res3
        print(f"completed train fit check for the '{MODEL_NAME[i]}' model")
        i +=1
    DF = pd.DataFrame(DF)
    return DF.to_csv('fit_classification_report.csv', index = False)

###############################################################################
# Check scores for each model on test data,save as csv & plot results
##############################################################################
def check_test_fit():
    test_full_df = load_experiment('test_full_df')
    test_from_fframe = load_experiment('test_from_fframe')
    test_from_CLframe = load_experiment('test_from_CLframe')
    DF = []
    i = 0
    for model in MODELS:
        print(f"checking test fit for the '{MODEL_NAME[i]}' model")
        # test fit for full data models
        mod1 = load_experiment(MODEL_NAME[i] + '_' + "best_model_f_full_df")
        test_res1 = list(map(classreport,test_full_df,mod1,[f"{MODEL_NAME[i]}_df_" + s for s in DATASET_NAMES]))
        test_res1 = reduce(operator.concat, test_res1)
        # test fit for full frame models 
        mod2 = load_experiment(MODEL_NAME[i] + '_' + "best_model_f_full_frame")
        test_res2 = list(map(classreport,test_from_fframe,mod2,[f"{MODEL_NAME[i]}_ff_" + s for s in DATASET_NAMES]))
        test_res2 = reduce(operator.concat, test_res2)
        # test fit for by class frame models 
        mod3 = load_experiment(MODEL_NAME[i] + '_' + "best_model_f_Class_frames")
        test_res3 = list(map(classreport,test_from_CLframe,mod3,[f"{MODEL_NAME[i]}_fCf_" + s for s in DATASET_NAMES]))
        test_res3 = reduce(operator.concat, test_res3)
        
        # Combine all together
        DF = DF + test_res1+test_res2+test_res3
        print(f"completed test fit check for the '{MODEL_NAME[i]}' model")
        i +=1
    DF = pd.DataFrame(DF)
    return DF.to_csv('test_classification_report.csv', index = False)

###############################################################################
# RUNS
###############################################################################    
print("Running experiments...")
#generate_data()
#calc_frame()
#train_test_split()
#fit_best_m()
check_train_fit()
check_test_fit()
print("Experiments successfully run")
