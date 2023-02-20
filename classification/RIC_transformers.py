#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

# feature extractors

import pycatch22

from kats.tsfeatures.tsfeatures import TsFeatures
from kats.consts import TimeSeriesData
tsFeatures = TsFeatures()

from tsfeatures import tsfeatures

import tsfel
# Retrieves a pre-defined feature configuration file to extract all available features
cfg = tsfel.get_features_by_domain()


from tsfresh import extract_features


# --------------------------------------

# ### Feature functions

# In[4]:


#########################################################
# functions to compute the features of the chosen extractor

def computeCatch22(d):
    ts = d.tolist()
    rawFeat = pycatch22.catch22_all(ts)
    
    # create a dictionary with the feature name as key and the value as value
    dictionary = {}
    for name,value in zip(rawFeat['names'],rawFeat['values']):
        dictionary[name] = [value]
         
    return pd.DataFrame.from_dict(dictionary)
    
    
def computeKats(d):
    ts = pd.DataFrame(d, columns = ['value'])
    rawFeatDict = TsFeatures().transform(ts)
        
    return pd.DataFrame.from_dict([rawFeatDict])
    
    
def computeTsfeatures(d):
    ts = pd.DataFrame(d, columns = ['y'])
    ts.insert(0, 'ds', pd.date_range(start='2020/12/01', periods=len(ts)))
    ts.insert(0, 'unique_id', len(ts) * [1])
    
    return tsfeatures(ts)

    
def computeTsfel(d):
    ts = pd.DataFrame(d, columns = ['value'])
                      
    return tsfel.time_series_features_extractor(cfg, ts)
    
    
def computeTsfresh(d):           
    ts = pd.DataFrame(d, columns = ['value'])
    ts[1] = len(ts) * [1]
    ts[2] = np.arange (1, len(ts)+1, 1.0)
    ts.columns = ['value', 'id', 'time']
                
    return  extract_features(ts, column_id='id', column_value = 'value', column_sort = 'time')


# create a switch which chooses the correct function depending on the chosen extractor
switch = {'catch22'   : computeCatch22, 
          'kats'      : computeKats, 
          'tsfeatures': computeTsfeatures, 
          'tsfel'     : computeTsfel, 
          'tsfresh'   : computeTsfresh}


# In[5]:


def getFeats(df, method):
    
    # stores the computed features
    frame = pd.DataFrame()
    nbrTS = len(df)

    # only for univariate time series
    for ts in range(nbrTS):
        
        tsArray = df[ts,0]
        features = switch[method](tsArray)
        
        frame = pd.concat([frame, features], axis=0, ignore_index=True)
        
        
    # replace infinite values by NaN
    #transformedFrame.replace([np.inf, -np.inf], np.nan, inplace=True)
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # remove features with more than 10% NaN
    transformedFrame = frame.drop(frame.columns[frame.apply(lambda col: (col.isnull().sum() / len(df)) > 0.1)], axis=1)
    
    # replace nan values by 0
    transformedFrame.replace([np.nan], 0, inplace=True)

        
    return transformedFrame


# ### Classes

# #### CATCH22

# In[6]:


class Catch22(BaseEstimator):
    
    def __init__(self):
        self.feats  = None
        self.labels = None
    
    # returns the class itself (shouldn't do anything on data --> fitting, the transformation is done)
    def fit(X, y):
        return Catch22
    
    # compute the features for all time series using the compute function
    def fit_transform(self, X, y):
        return getFeats(X, 'catch22')
    
    # compute the features for all time series using the compute function
    def _transform(self, X, y):
        return getFeats(X, 'catch22')
    
    


# #### KATS

# In[7]:


class Kats(BaseEstimator):
    
    def __init__(self):
        self.feats  = None
        self.labels = None
    
    # returns the class itself (shouldn't do anything on data --> fitting, the transformation is done)
    def fit(X, y):
        return Kats
    
    # compute the features for all time series using the compute function
    def fit_transform(self, X, y):
        return getFeats(X, 'kats')
    
    # compute the features for all time series using the compute function
    def _transform(self, X, y):
        return getFeats(X, 'kats')
    
    


# #### TSFEATURES

# In[8]:


class TSFeatures(BaseEstimator):
    
    def __init__(self):
        self.feats  = None
        self.labels = None
    
    # returns the class itself (shouldn't do anything on data --> fitting, the transformation is done)
    def fit(X, y):
        return TSFeatures
    
    # compute the features for all time series using the compute function
    def fit_transform(self, X, y):
        return getFeats(X, 'tsfeatures')
    
    # compute the features for all time series using the compute function
    def _transform(self, X, y):
        return getFeats(X, 'tsfeatures')
    
    


# #### TSFEL

# In[9]:


class TSFel(BaseEstimator):
    
    def __init__(self):
        self.feats  = None
        self.labels = None
    
    # returns the class itself (shouldn't do anything on data --> fitting, the transformation is done)
    def fit(X, y):
        return TSFel
    
    # compute the features for all time series using the compute function
    def fit_transform(self, X, y):
        return getFeats(X, 'tsfel')
    
    # compute the features for all time series using the compute function
    def _transform(self, X, y):
        return getFeats(X, 'tsfel')
    
    


# #### TSFRESH

# In[11]:


class TSFresh(BaseEstimator):
    
    def __init__(self):
        self.feats  = None
        self.labels = None
    
    # returns the class itself (shouldn't do anything on data --> fitting, the transformation is done)
    def fit(X, y):
        return TSFresh
    
    # compute the features for all time series using the compute function
    def fit_transform(self, X, y):
        extractedFeats = getFeats(X, 'tsfresh')
        return getFeats(X, 'tsfresh')
    
    # compute the features for all time series using the compute function
    def _transform(self, X, y):
        return getFeats(X, 'tsfresh')
    
    


# --------------------------------------
