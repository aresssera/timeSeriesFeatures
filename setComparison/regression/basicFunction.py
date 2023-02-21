#!/usr/bin/env python
# coding: utf-8

# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import glob
import os
import feather
sns.set(style="whitegrid")


# --------------------------------------

# ### Preparing the frames to store information

# In[8]:

#datasets = ['Adiac', 'Fish', 'OliveOil', 'Phoneme', 'ShapesAll', 'SwedishLeaf', 'WordSynonyms']

datasets = ['GlobalClimate', 'HumidityHouse',
            'HungaryChickenpox', 'IstanbulStockExchange', 
            'ParkingBirmingham', 'PedalMe']

#datasets =['GlobalClimate']

# row corresponds to classifier, col to dataset
coefOfDet = pd.DataFrame(columns=datasets)


# In[9]:

############## ANPASSEN
catch22    = pd.read_feather('data/output_catch22.feather')
kats       = pd.read_feather('data/output_kats.feather')
tsfeatures = pd.read_feather('data/output_tsfeatures.feather')
tsfel      = pd.read_feather('data/output_tsfel.feather')
tsfresh    = pd.read_feather('data/output_tsfresh.feather')

featF = [catch22, kats, tsfeatures, tsfel, tsfresh]

plainTimeSeries = []

for dataset in datasets:
    plainTimeSeries.append(pd.read_feather('data/' + dataset + '.feather')) #add data column as well?


# --------------------------------------

# In[19]:


def getAccuracyValues(reg):
    
    #--------
    # SKLEARN
    cod = []
    print('Round 1')
    
    for dataset, df in zip(datasets, plainTimeSeries):
    
        print(dataset)
    
        train = df[(df.data == dataset) & (df.set == 'train')].copy()
        X_train = train.iloc[:, :-4]
        y_train = train.iloc[:, -4]

        test = df[(df.data == dataset) & (df.set == 'test')].copy()
        X_test = test.iloc[:, :-4]
        y_test = test.iloc[:, -4]
    
        reg.fit(X_train, y_train)
    
        cod.append(reg.score(X_test, y_test))
    
    print('')
    coefOfDet.loc[0] = cod
    
    #-------------------
    # SKLEARN x FEATURES
    j = 1
    for frame in featF:
    
        cod = []
        print('Round ', j+1)
    
        for i,dataset in zip(range(len(datasets)), datasets):
        
            print(dataset)
        
            df = frame[frame.data == dataset].copy()
    
            train = df[(df.data == dataset) & (df.set == 'train')].copy()
            X_train = train.iloc[:, :-4]
            y_train = train.iloc[:, -4]

            test = df[(df.data == dataset) & (df.set == 'test')].copy()
            X_test = test.iloc[:, :-4]
            y_test = test.iloc[:, -4]
    
            reg.fit(X_train, y_train)
    
            cod.append(reg.score(X_test, y_test))

        coefOfDet.loc[j] = cod
        j += 1
        print('\n')
        
    return coefOfDet




# --------------------------------------

# ### Plotting error

# In[8]:


def plotPerDataset(axes, y_pos, nbrPlots, df, regressors):
    
    for c in range(nbrPlots):
        
        axes[c].axhline(y=0, color='gray')
        axes[c].set_title(df.columns[c])
        
        # Create bars
        color = ['black'] + (len(df)-1)*['maroon']
        axes[c].bar(y_pos, df.iloc[:, c], color=color)
        
        axes[c].set_xticks(y_pos, regressors, rotation='vertical')
        axes[c].title.set_size(15)
    
    
def plotPerClassifier(axes, y_pos, nbrPlots, df, regressors):
    
    for c in range(nbrPlots):
        
        axes[c].axhline(y=0, color='gray')
        axes[c].set_title(regressors[c], size=20)
        
        color = 'maroon'
        
        if c == 0:
            color = 'black'
        
        # Create bars
        axes[c].bar(y_pos, df.iloc[c], color=color)
        
        axes[c].set_xticks(y_pos, datasets, rotation='vertical') 
        axes[c].title.set_size(15)
        
        
        
    


# In[21]:


def plotError(df, regressors, dateiName, perDataset = True):
    
    cols = 3
    
    if perDataset:
        nbrPlots = len(df.columns)
        y_pos    = np.arange(len(df))
    else:
        nbrPlots = len(df)
        y_pos    = np.arange(len(df.columns))
    
    rows = int(np.ceil(nbrPlots/cols))
    
    fig, ax = plt.subplots(rows, cols, figsize=(30, 30))
    plt.setp(ax, ylim=(0,1))
    
    # using padding
    fig.tight_layout(pad=14.0)
    
    
    axes = ax.flatten()
    fs = 40
    
    if perDataset:
        plotPerDataset(axes, y_pos, nbrPlots, df, regressors)
        fig.text(0.5, 0.02, 'Regressor', ha='center', fontsize=fs)
        fig.text(0.04, 0.5, 'Coefficient of Determination', va='center', rotation='vertical', fontsize=fs)
    else:
        plotPerClassifier(axes, y_pos, nbrPlots, df, regressors)
        fig.text(0.5, 0.02, 'Dataset', ha='center', fontsize=fs)
        fig.text(0.04, 0.5, 'Coefficient of Determination', va='center', rotation='vertical', fontsize=fs)
        
    fig.savefig(dateiName + '.pdf', bbox_inches='tight') 

# In[ ]:




