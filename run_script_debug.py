import pandas as pd
# Fit models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# Helper functions
from Experiments.generateData import generateData, frame
from Experiments.helperfuncs import save_experiment,load_experiment
from Experiments.helperfuncs import Gridsearch,classreport
from Experiments.helperfuncs import get_auc_scores,plot_auc
from Experiments.helperfuncs import train_test_split

###############################################################################
# load data if already generated otherwise generate and save data
###############################################################################
DATASETS = ["5000_70_30_25d"]

PARAMS = [[5000,[.70,.30],25]]

MODELS = [LogisticRegression(),
          SVC(),
          RandomForestClassifier(),
          XGBClassifier()]

MODEL_NAME = ['logisticReg','SVM','RF','GB_DT']

M_PARAMS = [
{'C': [0.1,0.5,1,10,50,100], 'max_iter': [250],'fit_intercept':[True],
 'intercept_scaling':[1],'penalty':['l2'], 'tol':[0.0001], 'solver':['lbfgs']},

{'C': [0.1,0.5,1,10,50,100], 'gamma': [0.1,0.01,0.001], 'probability':[True]},

{'max_depth': [3, 5, 6, 7], 'max_features': [2,4,5],
'min_samples_split': [3, 5, 6, 7], 'n_estimators':[10,50,100]},

 {'max_depth': [5,6,7,8], 'gamma': [0.1, 0.01, 0.001], 
              'learning_rate': [0.05,0.1, 0.2, 0.3]}]
###############################################################################
# Generate and save the various datasets
###############################################################################
def generate_data():
    i = 0
    for params in PARAMS:
        print(f"generating dataset for '{params}'")
        data = generateData(n=params[0],d=params[2], classWeights = params[1])
        save_experiment(data,DATASETS[i])
        i += 1
        print(f"Completed dataset generation for '{params}'")
    return None

###############################################################################
# Load the generated datasets, compute the frames and save the frame indices
###############################################################################
def calc_frame():
    for dataset in DATASETS:
        print(f"entering calculation of frame for '{dataset}'")
        data = load_experiment(dataset)
        X= data[:,1:]
        q = frame(X)
        save_experiment(q, dataset + "_frame")
        print(f"exiting calculating frame for '{dataset}'")
    return None

###############################################################################
# Carry out grid search to get best model for each dataset and save best params
###############################################################################
def grid_search():
    for dataset in DATASETS:
        data1 = train_test_split(dataset,dtype = 'dataset')
        train1 = data1[0]
        X,y = train1[:,1:],train1[:,0]
        
        data2 = train_test_split(dataset,dtype = 'frame')
        train2 = data2[0]
        X_frame,y_frame = train2[:,1:],train2[:,0]
        i = 0
        for model in MODELS:
            print(f"entering grid search for the '{MODEL_NAME[i]} for '{dataset}'")
            Gridsearch(model,M_PARAMS[i],dataset,X,y,modelName = MODEL_NAME[i])
            Gridsearch(model,M_PARAMS[i],dataset+'_frame',X_frame,y_frame,modelName = MODEL_NAME[i])
            print(f"exiting grid search for the '{MODEL_NAME[i]} for '{dataset}'")
            i +=1
    return None
###############################################################################
# Fit best model for each dataset and save model
###############################################################################
def fit_best_m():
    for dataset in DATASETS:
        data1 = train_test_split(dataset,dtype = 'dataset')
        train1 = data1[0]
        X,y = train1[:,1:],train1[:,0]
        
        data2 = train_test_split(dataset,dtype = 'frame')
        train2 = data2[0]
        X_frame,y_frame = train2[:,1:],train2[:,0]
        i=0
        for model in MODELS:
            print(f"fitting best '{MODEL_NAME[i]}' model for '{dataset}'")
            best_params = load_experiment(MODEL_NAME[i] + '_' + dataset + "_best_params")
            best_params_f = load_experiment(MODEL_NAME[i] + '_' + dataset + "_frame"+"_best_params")
            mod1 = model.set_params(**best_params)
            mod1 = mod1.fit(X,y)
            mod2 = model.set_params(**best_params_f)
            mod2 = mod2.fit(X_frame,y_frame)
            save_experiment(mod1,f"best_'{MODEL_NAME[i]}'_'{dataset}'")
            save_experiment(mod2,f"best_'{MODEL_NAME[i]}'_'{dataset}'_frame")
            print(f"Completed best '{MODEL_NAME[i]}' model for '{dataset}'")
            i +=1
    return None

###############################################################################
# Check fit scores for each model for each dataset and save as csv
###############################################################################
def check_train_fit():
    df = []
    for dataset in DATASETS:
        FPR = []
        TPR = []
        AUC = []
        PLOTNAMES = []
        data1 = train_test_split(dataset,dtype = 'dataset')
        train1 = data1[0]
        X,y = train1[:,1:],train1[:,0]
        data2 = train_test_split(dataset,dtype = 'frame')
        train2 = data2[0]
        X_frame,y_frame = train2[:,1:],train2[:,0]
        i = 0
        for model in MODELS:
            print(f"checking train fit for the '{MODEL_NAME[i]}' for '{dataset}'")
            mod = load_experiment(f"best_'{MODEL_NAME[i]}'_'{dataset}'")
            mod_fr = load_experiment(f"best_'{MODEL_NAME[i]}'_'{dataset}'_frame")
            df = df + classreport(X,y,mod,f"'{MODEL_NAME[i]}'_'{dataset}'")
            df = df + classreport(X_frame,y_frame,mod_fr,f"'{MODEL_NAME[i]}'_'{dataset}'_frame")
            
            auc_score, fpr_df, tpr_df = get_auc_scores(X,y, mod)
            auc_score_f, fpr_df_f, tpr_df_f = get_auc_scores(X_frame,y_frame, mod_fr)
            
            AUC.append(auc_score)
            AUC.append(auc_score_f)
            
            FPR.append(fpr_df)
            FPR.append(fpr_df_f)
            
            TPR.append(tpr_df)
            TPR.append(tpr_df_f)
            
            PLOTNAMES.append(f"'{MODEL_NAME[i]}'_'{dataset}'")
            PLOTNAMES.append(f"'{MODEL_NAME[i]}'_'{dataset}'_frame")
            print(f"completed train fit check for the '{MODEL_NAME[i]}' model for '{dataset}'")
            i +=1
        plot_auc(FPR,TPR,AUC,PLOTNAMES,dataset+'_train_set_score')
    df = pd.DataFrame(df)
    return df.to_csv('fit_classification_report.csv', index = False)
 
###############################################################################
# Check scores for each model on test data,save as csv & plot results
###############################################################################
def check_test_fit():
    df = []
    for dataset in DATASETS:
        FPR = []
        TPR = []
        AUC = []
        PLOTNAMES = []
        data1 = train_test_split(dataset,dtype = 'dataset')
        test1 = data1[1]
        X,y = test1[:,1:],test1[:,0]
        
        data2 = train_test_split(dataset,dtype = 'frame')
        test2 = data2[1]
        X_frame,y_frame = test2[:,1:],test2[:,0]
        i = 0
        for model in MODELS:
            print(f"checking test fit for the '{MODEL_NAME[i]}' for '{dataset}'")
            mod = load_experiment(f"best_'{MODEL_NAME[i]}'_'{dataset}'")
            mod_fr = load_experiment(f"best_'{MODEL_NAME[i]}'_'{dataset}'_frame")
            df = df + classreport(X,y,mod,f"'{MODEL_NAME[i]}'_'{dataset}'")
            df = df + classreport(X_frame,y_frame,mod_fr,f"'{MODEL_NAME[i]}'_'{dataset}'_frame")
            
            auc_score, fpr_df, tpr_df = get_auc_scores(X,y, mod)
            auc_score_f, fpr_df_f, tpr_df_f = get_auc_scores(X_frame,y_frame, mod_fr)
            
            AUC.append(auc_score)
            AUC.append(auc_score_f)
            
            FPR.append(fpr_df)
            FPR.append(fpr_df_f)
            
            TPR.append(tpr_df)
            TPR.append(tpr_df_f)
            
            PLOTNAMES.append(f"'{MODEL_NAME[i]}'_'{dataset}'")
            PLOTNAMES.append(f"'{MODEL_NAME[i]}'_'{dataset}'_frame")
            print(f"completed test fit check for the '{MODEL_NAME[i]}' model for '{dataset}'")
            i += 1
        plot_auc(FPR,TPR,AUC,PLOTNAMES,dataset+'_test_set_score')
    df = pd.DataFrame(df)
    return df.to_csv('prediction_classification_report.csv', index = False)
###############################################################################
###############################################################################
# RUNS
###############################################################################    
generate_data()
calc_frame()
grid_search()
fit_best_m()
check_train_fit()
check_test_fit()