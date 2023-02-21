import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

datasets = ['Adiac', 'Fish', 'OliveOil', 'Phoneme', 'ShapesAll', 'SwedishLeaf', 'WordSynonyms']

catch22    = pd.read_feather('featureFrames/output_catch22.feather')
kats       = pd.read_feather('featureFrames/output_kats.feather')
tsfeatures = pd.read_feather('featureFrames/output_tsfeatures.feather')
tsfel      = pd.read_feather('featureFrames/output_tsfel.feather')
tsfresh    = pd.read_feather('featureFrames/output_tsfresh.feather')

extractors = [catch22,
              kats,
              tsfeatures,
              tsfel,
              tsfresh]


def plotPerDataset(axes, y_pos, nbrPlots, df):
    
    for c in range(nbrPlots):
        
        axes[c].set_title(df.columns[c])
        axes[c].set_xlabel('Classifier')
        axes[c].set_ylabel('Accuracy')
        
        # Create bars
        axes[c].bar(y_pos, df.iloc[:, c], color='maroon')
        
        #
        axes[c].set_xticks(y_pos, classifiers, rotation='vertical')
    
    
def plotPerClassifier(axes, y_pos, nbrPlots, df):
    
    for c in range(nbrPlots):
        
        axes[c].set_title(classifiers[c])
        axes[c].set_xlabel('Dataset')
        axes[c].set_ylabel('Accuracy')
        
        # Create bars
        axes[c].bar(y_pos, df.iloc[c], color='maroon')
        
        #
        axes[c].set_xticks(y_pos, datasets, rotation='vertical') 
    
    
def plotAccuracy(df, perDataset = True):
    
    cols = 3
    
    if perDataset:
        nbrPlots = len(df.columns)
        y_pos    = np.arange(len(df))
    else:
        nbrPlots = len(df)
        y_pos    = np.arange(len(df.columns))
    
    rows = int(np.ceil(nbrPlots/cols))
    
    fig, ax = plt.subplots(rows, cols, figsize=(30, 30))
    # using padding
    fig.tight_layout(pad=14.0)
    
    axes = ax.flatten()
    
    if perDataset:
        plotPerDataset(axes, y_pos, nbrPlots, df)
    else:
        plotPerClassifier(axes, y_pos, nbrPlots, df)
    
     