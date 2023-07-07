import warnings

import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV 
#from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
)
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, f1_score, recall_score

import pickle

from src.preprocessing import scale_method 

warnings.filterwarnings("ignore")


class_weights = {0: 1, 1: 10}

models_dict = {
    "Random_Forest" : RandomForestClassifier(class_weight=class_weights),
    "XGBoost" : XGBClassifier(),
    "SVM" : SVC(class_weight=class_weights),
    "AdaBoost": AdaBoostClassifier(),
    "LR": LogisticRegression(class_weight=class_weights )  
}


def train(X, y, args):
    list_results = []
    if args.use_cv == "True":
        #Create stratified K-folds
        cv = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)    
        # iterate over classifiers
        if args.model_name == 'all':
            df = pd.DataFrame()
            for name, clf in models_dict.items():
                list_results = cv_training(name,clf,X,y,cv,list_results,args)
            
        else: 
            name = args.model_name
            clf = models_dict[name]
            list_results = cv_training(name,clf,X,y,cv,list_results,args)
        
        df = pd.DataFrame(
                        list_results,
                        columns=["Classifier", f"Mean {args.metric} over 8 folds", "Standard Deviation"]
                        )    
            
    else:
        #if args.hp_gridsearch == False:
        if args.model_name == 'all':
            df = pd.DataFrame()
            for name, clf in models_dict.items():
                list_results = no_cv_training(name,clf,X,y,list_results,args)
                print(f"Model {name} trained - {list_results} ")

        else:
            name = args.model_name
            clf = models_dict[name]
            list_results = no_cv_training(name,clf,X,y,list_results,args)
            
        df = pd.DataFrame(list_results, columns=["Classifier", args.metric])
            
# -       else :
#             if args.model_name == 'XGBoost':
#                 name = args.model_name
#                 clf = models_dict[name]
#                 hyperparameter_training(clf,X,y,args)
#                 df = pd.DataFrame(['XGBoost'], columns=["Classifier", args.metric]) 


    
    # save results in a csv file       
    save_results(df,args)
    return df 
    

#def get_metric(args):
#    if args.metric == "f1":
#        scoring = {'f1': make_scorer(f1_score)}
#   if args.metric == "recall":
#        scoring = {'recall': make_scorer(recall_score)}
#        return scoring
    
def cv_training(name,clf,X,y,cv,list_results,args):
    """ Function to train a model with cross-validation"""
    #scoring = get_metric(args)
    scaler = scale_method(args)
    pipeline = make_pipeline(scaler,clf)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=args.metric)
    list_results.append([name, scores.mean(), scores.std()])
    
    return list_results
    
def no_cv_training(name,clf,X,y,list_results,args):
    """ Function to train a model without cross-validation"""
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=0
        )
    scaler = scale_method(args)
    pipeline = make_pipeline(scaler,clf)
    if name != 'XGBoost':
        pipeline.fit(X_train, y_train)
    else:
        sample_weights = [1 if label == 0 else 10 for label in y_train]
        pipeline.fit(X_train, y_train, xgbclassifier__sample_weight=sample_weights)
 

    y_pred = pipeline.predict(X_test)
    if args.metric == 'f1':
        score = f1_score(y_test, y_pred)
    if args.metric == 'recall':
        score = recall_score(y_test, y_pred)
    if score > 0.8:
        save_model(name,pipeline,args)
    list_results.append([name, score])
    
    return list_results
    
def base_name(model_name,args):
    """ Base name used for result and model filenames"""
    if args.use_cv == True:
        cv = 'cv_'
    else:
        cv = ''
    if args.scaler != '':
        scaler = args.scaler + '_'
    else:
        scaler = ''
    base_name = str(model_name + '_' + scaler + cv + args.metric)
    return(base_name)
    
def save_results(df,args):
    """ Save the performance of a model after training"""
    filename = base_name(args.model_name, args) + '.csv'
    df.to_csv(f'./results/{filename}',index=False)
    

def save_model(name,model,args) :
    """ Save a model in .pkl format"""
    print(base_name(name, args)) 
    filename = base_name(name, args) + '.pkl'
    with open(f'./models/{filename}', 'wb') as file:
        pickle.dump(model, file)
    
    
    
# def hyperparameter_training(clf,X,y,args):
#     # Define the hyperparameters to be tuned 
#     hyperparameters = {
#     'gradientboostingclassifier__learning_rate': [0.1, 0.01, 0.001],
#     'gradientboostingclassifier__max_depth': [3, 5, 7],
#     'gradientboostingclassifier__n_estimators': [100, 200, 300]
#     }   

#     X_train, X_test, y_train, y_test = train_test_split(
#                 X, y, test_size=0.20, random_state=0
#             )

#     scaler = scale_method(args)
#     pipeline = make_pipeline(scaler,clf)

#     # Perform grid search cross-validation
#     grid_search = GridSearchCV(pipeline, hyperparameters, cv=5)
#     grid_search.fit(X_train, y_train)

#     # Get the best model and its hyperparameters
#     best_model = grid_search.best_estimator_
#     best_params = grid_search.best_params_

#     # Evaluate the best model on the test set
#     y_pred = best_model.predict(X_test)
#     auc_roc_score = roc_auc_score(y_test, y_pred)
#     print("Best model auc:", auc_roc_score)
#     print("Best hyperparameters:", best_params)
#     save_model('best_hyp',best_model,args)