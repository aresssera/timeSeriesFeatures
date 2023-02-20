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
import feather
sns.set(style="whitegrid")


# --------------------------------------

# In[11]:


def getOneFrame(dataset, labels, predictions1, predictions2):
    
    # stores series 
    finalFrame = pd.DataFrame()
    
    for row in dataset.iterrows():
        finalFrame = pd.concat([finalFrame, row[1].values[0].to_frame().T], ignore_index=True)
    
    # adds labels at the end of the frame
    finalFrame['label'] = [int(i) for i in labels]
    finalFrame['pred1']  = [int(i) for i in predictions1]
    finalFrame['pred2']  = [int(i) for i in predictions2]

    return finalFrame


# In[20]:


def plotTsClasses(df, axes, colors, predX, i, c):
    classFrame = df[df['label'] == (c)]
            
    axes[i].set_title('Time series type ' + str(c))
    axes[i].set_xlabel('Time')
    axes[i].set_ylabel('Value')
        
    for j in range(len(classFrame)):
            
        ts = classFrame.iloc[j]
            
        if ts['label'] != ts[predX]:
            axes[i].plot(range(len(ts)-3), ts[:-3], color=colors[int(ts[predX])-1], linewidth=6, linestyle='dashdot')
        else:
            axes[i].plot(range(len(ts)-3), ts[:-3], color=colors[int(ts[predX])-1], linewidth=4)


# In[33]:


def plotSeriesPerClass(dataset, labels, predictions1, predictions2, title):
    
    df = getOneFrame(dataset, labels, predictions1, predictions2)
    
    classes = df['label'].unique()
    nbrClasses = len(classes)
    cols = 2
    #rows = len(predictions1)
    
    fig, ax = plt.subplots(nbrClasses, cols, figsize=(30, 70))
    # using padding
    fig.tight_layout(pad=7.0)
    fig.suptitle(title)
    
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
    
    # first for the left side
    for i, c in zip(range(0, (2*nbrClasses)+1, 2), classes):
        plotTsClasses(df, axes, colors, 'pred1', i, c)
        
    # then for the right side:
    for i, c in zip(range(1, (2*nbrClasses)+1, 2), classes):
        plotTsClasses(df, axes, colors, 'pred2', i, c)
                


# In[34]:


plotSeriesPerClass(X_test, y_test, y_predict1, y_predict2, 'catch22 x ' + names[8])


# --------------------------------------
