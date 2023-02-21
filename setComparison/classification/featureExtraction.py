#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd


# In[2]:


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

# ### CREATE CLASSES

# In[8]:


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
    ts = d.to_frame()
    ts.columns = ['value']
    rawFeatDict = TsFeatures().transform(ts)
        
    return pd.DataFrame.from_dict([rawFeatDict])
    
    
def computeTsfeatures(d):
    ts = pd.DataFrame()
    ts['y']=  d.to_frame()
    ts.insert(0, 'ds', pd.date_range(start='2020/12/01', periods=len(ts)))
    ts.insert(0, 'unique_id', len(ts) * [1])
    
    return tsfeatures(ts)

    
def computeTsfel(d):
    ts = d.to_frame()
                      
    return tsfel.time_series_features_extractor(cfg, ts)
    
    
def computeTsfresh(d):           
    ts = d.to_frame()
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


# In[3]:


def getFeats(df, method):
    
    # stores the computed features
    transformedFrame = pd.DataFrame()
    
    colNames = df.columns
    nbrCols  = len(colNames)
    nbrTS    = len(df)

    for col in range(nbrCols):
        dim = colNames[col]
        currentDim = pd.DataFrame()
        
        for ts in range(nbrTS):
            
            features = switch[method](df.iloc[ts].values[0])
            currentDim = pd.concat([currentDim, features], axis=0, ignore_index=True)
        
        currentDim.columns = [dim + '_' + originalName for originalName in currentDim.columns]
        transformedFrame = pd.concat([transformedFrame, currentDim], axis=1, ignore_index=False)
        
        # replace infinite values by NaN
        transformedFrame.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        return transformedFrame


# #### CATCH22

# In[10]:


class Catch22:
    
    # returns the class itself (shouldn't do anything on data --> fitting, the transformation is done)
    def fit(X, y):
        return Catch22
    
    # compute the features for all time series using the compute function
    def transform(df):
        return getFeats(df, 'catch22')
    
    


# #### KATS

# In[11]:


class Kats:
    
    # returns the class itself (shouldn't do anything on data --> fitting, the transformation is done)
    def fit(X, y):
        return Kats
    
    # compute the features for all time series using the compute function
    def transform(df):
        return getFeats(df, 'kats')
    


# #### TSFEATURES

# In[20]:


class TSFeatures:
    
    # returns the class itself (shouldn't do anything on data --> fitting, the transformation is done)
    def fit(X, y):
        return TSFeatures
    
    # compute the features for all time series using the compute function
    def transform(df):
        return getFeats(df, 'tsfeatures')


# #### TSFEL

# In[14]:


class TSFel:
    # returns the class itself (shouldn't do anything on data --> fitting, the transformation is done)
    def fit(X, y):
        return TSFel
    
    # compute the features for all time series using the compute function
    def transform(df):
        return getFeats(df, 'tsfel')


# #### TSFRESH

# In[15]:


class TSFresh:
    
    # returns the class itself (shouldn't do anything on data --> fitting, the transformation is done)
    def fit(X, y):
        return TSFresh
    
    # compute the features for all time series using the compute function
    def transform(df):
        return getFeats(df, 'tsfresh')

