#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import glob
import os
import sklearn
import sktime
from sktime.datasets import load_from_tsfile_to_dataframe
sns.set(style="whitegrid")


# In[125]:


#from featureExtractionClasses import *


# --------------------------------------

# ### LOAD DATA

# In[160]:


# SWEDISH LEAF

# load training data
X_train, y_train = load_from_tsfile_to_dataframe('data/SwedishLeaf_TRAIN.ts')

# load test data
X_test, y_test = load_from_tsfile_to_dataframe('data/SwedishLeaf_TEST.ts')

X_train


# In[180]:


'''
# OLIVE OIL

# load training data
X_train, y_train = load_from_tsfile_to_dataframe('data/OliveOil_TRAIN.ts')

# load test data
X_test, y_test = load_from_tsfile_to_dataframe('data/OliveOil_TEST.ts')

X_train
'''


# In[130]:


def getOneFrame(dataset, labels, predictions):
    
    # stores series 
    finalFrame = pd.DataFrame()
    
    for row in dataset.iterrows():
        finalFrame = pd.concat([finalFrame, row[1].values[0].to_frame().T], ignore_index=True)
    
    # adds labels at the end of the frame
    finalFrame['label'] = [int(i) for i in labels]
    finalFrame['pred']  = [int(i) for i in predictions]

    return finalFrame


# In[175]:


def plotSeriesPerClass(dataset, labels, predictions):
    
    df = getOneFrame(dataset, labels, predictions)
    
    nbrClasses = len(df['pred'].unique())
    cols = 2
    rows = int(np.ceil(nbrClasses/cols))
    
    fig, ax = plt.subplots(rows, cols, figsize=(30, 30))
    # using padding
    fig.tight_layout(pad=7.0)
    
    # colours
    #----------------------------------
    colors = ['darkmagenta', 
              'darkkhaki', 
              'cyan', 
              'slategrey',
              'tomato', 
              'navy',
              'mediumaquamarine', 
              'palegreen',
              'fuchsia',
              'darkgreen',
              'orange',
              'orchid',
              'dimgrey',
              'darksalmon',
              'rebeccapurple',
              'cadetblue']
    #----------------------------------
    
    
    axes = ax.flatten()
    for c in range(nbrClasses):
        classFrame = df[df['label'] == (c+1)]
        axes[c].set_title('Time series type ' + str(c+1))
        axes[c].set_xlabel('Time')
        axes[c].set_ylabel('Value')
        
        for i in range(len(classFrame)):
            
            ts = classFrame.iloc[i]
            
            if ts['label'] != ts['pred']:
                axes[c].plot(range(len(ts)-2), ts[:-2], color=colors[int(ts['pred'])-1], linewidth=6, linestyle='dashdot')
            else:
                axes[c].plot(range(len(ts)-2), ts[:-2], color=colors[int(ts['pred'])-1], linewidth=4)
                
            


# In[176]:


plotSeriesPerClass(X_train, y_train, y_train)


# In[177]:


'''
def plotTimeSeries(df, label):
    
    nbrTS = len(df)
    
    cols = 2
    rows = int(np.ceil(nbrTS/cols))
    
    fig, ax = plt.subplots(rows, cols, figsize=(30, 30))
    # using padding
    fig.tight_layout(pad=7.0)
    
    
    # colours
    #----------------------------------
    dz = np.asarray([*range(0, len(label)+1, 1)])

    norm = plt.Normalize()
    colors = plt.cm.jet(norm(dz))

    lower = dz.min()
    upper = dz.max()
    colors = plt.cm.jet((dz-lower)/(upper-lower))

    def get_colors(inp, colormap, vmin=None, vmax=None):
        norm = plt.Normalize(vmin, vmax)
        return colormap(norm(inp))

    colors = get_colors(dz, plt.cm.jet)
    
    #----------------------------------
    
    axes = ax.flatten()
    for row, a in zip(range(nbrTS), axes):
        
        ts = seriesToNumpy(df.iloc[row])
        
        a.plot(range(len(ts)), ts, color=colors[int(label[row])],linewidth=4)
        a.set_title('Time series type ' + str(label[row]))
        a.set_xlabel('Time')
        a.set_ylabel('Value')


#plotTimeSeries(X_train.iloc[0:20], y_train[0:20])
'''


# --------------------------------------

# In[178]:


from sktime.classification.feature_based import Catch22Classifier
classifier = Catch22Classifier()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
classifier.score(X_test, y_test)


# In[179]:


plotSeriesPerClass(X_test, y_test, y_pred)

