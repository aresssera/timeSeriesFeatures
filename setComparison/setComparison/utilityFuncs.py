import numpy as np
import pandas as pd
import scipy.stats as stats

import pycatch22
from kats.tsfeatures.tsfeatures import TsFeatures
from kats.consts import TimeSeriesData
tsFeatures = TsFeatures()
from tsfeatures import tsfeatures
import tsfel
cfg = tsfel.get_features_by_domain()
from tsfresh import extract_features


# ---------------------------------------------------------------------------------------------


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


# In[99]:


def getFeats(df, method):
    
    # stores the computed features
    frame = pd.DataFrame()
    
    #print('shape = ', df.shape)
    
    nbrTS    = len(df)
    print('--------------------------------------------------')
    print(len(df[0,0]))# is the length of the interval
    print(df[0,0])

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


# ---------------------------------------------------------------------------------------------


# takes the columns of a dataframe as the rows
def getFeatureRows(features, i, group, extractor):
    
    df = pd.DataFrame(columns=['id', 'group', 'names', 'values', 'method'])

    for column in features:
        df2 = {'id': i, 'group': group, 'names': column, 'values': features[column].values, 'method': extractor}
        df = pd.concat([df, pd.DataFrame.from_dict(df2)], ignore_index=True)
        
    return df


# Calculates all features from a chosen feature set for all time series' in the dataset (df).


def compCatch22(d, i, time_var, values_var, group):
    
    ts = dataFrameToList(d[values_var].to_frame())
    rawFeat = pycatch22.catch22_all(ts)
    
    # create a dictionary with the feature name as key and the value as value
    dictionary = {}
    for name,value in zip(rawFeat['names'],rawFeat['values']):
        dictionary[name] = [value]
        
    # then create a dataframe, and from that a dataframe row per feature
    features = pd.DataFrame.from_dict(dictionary)
    return getFeatureRows(features, i, group, 'catch22')
    
    
def compKats(d, i, time_var, values_var, group):
    
    rawFeatDict = TsFeatures().transform(d)
        
    # then create a dataframe, and from that a dataframe row per feature
    features = pd.DataFrame.from_dict([rawFeatDict])
    return getFeatureRows(features, i, group, 'kats')
    

def compTsfeatures(d, i, time_var, values_var, group):
    
    ts = d[values_var].to_frame()
    
    ts.rename(columns={values_var: "y"}, inplace=True)
    ts.insert(0, 'ds', pd.date_range(start='2020/12/01', periods=len(ts)))
    ts.insert(0, 'unique_id', len(ts) * [i])
    
    features = tsfeatures(ts)
    return getFeatureRows(features, i, group, 'tsfeatures')
    
    
def compTsfel(d, i, time_var, values_var, group):
    
    ts = d[values_var].to_frame()
    
    features = tsfel.time_series_features_extractor(cfg, ts)
    return getFeatureRows(features, i, group, 'tsfel')
    
    
def compTsfresh(d, i, time_var, values_var, group):
    features = extract_features(d, column_id='id', column_value = values_var, column_sort = time_var)
    return getFeatureRows(features, i, group, 'tsfresh')


# create a switch which chooses the correct function depending on the chosen extractor
switch2 = {'catch22' : compCatch22, 'kats' : compKats, 'tsfeatures': compTsfeatures, 'tsfel' : compTsfel, 'tsfresh' : compTsfresh}



# calculates all features for all time series' of the df
def calculate_features(df, id_var, time_var, values_var, group_var, feature_set):
    
    calculatedFeatures = pd.DataFrame()
    
    for i in df['id'].unique():
        
        print("Computing features for ", i)
        # d as all the data available for the current time series
        d = df.loc[df[id_var] == i]
        group = d[group_var].unique()[0]
        computeFeat = switch2[feature_set](d, i, time_var, values_var, group)
        calculatedFeatures = pd.concat([calculatedFeatures, computeFeat], ignore_index=True)
        
    return calculatedFeatures.sort_values(['names', 'id'], ascending=[False, True], inplace=False, ignore_index=True)
        

# ---------------------------------------------------------------------------------------------


class Catch22():
    
    # calculates the features for 1 time series X
    def calcFeats(X):
        return switch['catch22'](X)
    

class Kats():
    
    # calculates the features for 1 time series X
    def calcFeats(X):
        return switch['kats'](X)
    
    
class TSFeatures():
    
    # calculates the features for 1 time series X
    def calcFeats(X):
        return switch['tsfeatures'](X)
    
    
class TSFel():
    
    # calculates the features for 1 time series X
    def calcFeats(X):
        return switch['tsfel'](X)



class TSFresh():
    
    # calculates the features for 1 time series X
    def calcFeats(X):
        return switch['tsfresh'](X)
    
    
# ---------------------------------------------------------------------------------------------
# used for computing the correlation matrices

def zScore(df):
    df['values'] = stats.zscore(df['values'].values, nan_policy='omit')
    return df


# returns a dictionary with all the good features
def get_problematic_features(d):
    
    goodFeats = {}
    
    for i in d['method'].unique():
        
        methodFrame = d.loc[d['method'] == i]
        
        # replace infinite values by NaN
        methodFrame.replace([np.inf, -np.inf], np.nan, inplace=True)
        totNumFeats = len(methodFrame['names'].unique())
        
        # the values need to be normed with z-score (grouped by feature name)
        methodFrame = methodFrame.groupby('names').apply(zScore)
        
        tmpF = pd.pivot_table(methodFrame, index=['id'], columns=['names'])
        
        # checks if more than 90% are NaN
        gf = tmpF.drop(tmpF.columns[tmpF.apply(lambda col: (col.isnull().sum() / len(tmpF)) > 0.1)], axis=1)
        
        print(f"# of bad features for {i:10} = {totNumFeats:4} - {len(gf.columns):4}")
        
        # store the names of the good features
        goodFeats[i] = [i[1] for i in gf.columns.values]
    
    print('')
    
    return goodFeats


# returns a list with the ID's of all the 'good' time series' and the filtered version of the raw dataset
def remove_problematic_datasets(d):
    
    # returns a dictionary with a list of good features for each method
    goodFeats = get_problematic_features(d)
    
    # removes infinite values
    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    #---------------------------------------------
    # remove the bad features
   
    filtered = pd.DataFrame()
    
    for method in d['method'].unique():
        
        # remove all the features that are not mentioned
        temp = d.loc[(d['method'] == method) & (d['names'].isin(goodFeats[method]))]
        filtered = pd.concat([filtered, temp], ignore_index=True)
    
    #----------------------------------------------------------------------------------------
    # now that the bad features are removed, we need to remove any time series that has a NaN
    
    badIDs = []
    
    for method in filtered['method'].unique():
        
        # normalizing the values
        methodFrame = filtered.loc[filtered['method'] == method].groupby('names').apply(zScore)
        
        # rows are time series', columns are features
        tmpFrame = pd.pivot_table(methodFrame, index=['id'], columns=['names'])
        badIDs = badIDs + np.where(tmpFrame.isna().any(axis=1))[0].tolist()
    
    badIDs = list(set(badIDs))
    
    #print(f'# of bad IDs = {len(badIDs):3} of {len(d['id'].unique()):4}')
    
    # this will store all the good features and all the good id's in the dataframe 'filtered'
    filtered = filtered.loc[~filtered['id'].isin(badIDs)]
    # Change the row indexes
    filtered.index = list(range(len(filtered)))
    
    goodIDs = filtered['id'].unique()
    
    return filtered, goodIDs

# Adds a column called comb_id which stores the method name combined with the feature name.
def combine_method_and_name(row):
    return row['method'] + '_' + row['names']