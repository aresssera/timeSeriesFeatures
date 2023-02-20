get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import glob
import os
import feather
sns.set(style="whitegrid")
import shap
from sklearn.ensemble import RandomForestRegressor

datasets = ['GlobalClimate', 'HumidityHouse',
            'HungaryChickenpox', 'IstanbulStockExchange', 
            'ParkingBirmingham', 'PedalMe']


def computeSHAP(X_train, y_train, dataset):
    
    # Prepares a default instance of the random forest classifier
    regressor = RandomForestRegressor(random_state=3)
    
    X = X_train.iloc[0:2]
    y = y_train[0:2]
    
        
    # Fits the model on the data
    regressor.fit(X, y)
    
    #print(dataset, 'score:', regressor.score(X_test, y_test))
    print(dataset)


    # Create object that can calculate shap values
    explainer = shap.TreeExplainer(regressor)
    # Calculate Shap values
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, plot_type="bar")


def getSets(dataset, ex):
    
    train = ex.loc[(ex['data'] == dataset) & (ex['set'] == 'train')].reset_index().iloc[:,1:]
    test  = ex.loc[(ex['data'] == dataset) & (ex['set'] == 'test')].reset_index().iloc[:,1:]

    X_train = train.iloc[:,:-4]
    y_train = train['y']

    X_test = test.iloc[:,:-4]
    y_test = test['y']
    
    return X_train, y_train, X_test, y_test

