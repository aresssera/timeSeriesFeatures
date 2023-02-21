import seaborn as sns
import pandas as pd


datasets = ['Adiac', 'Fish', 'OliveOil', 'Phoneme', 'ShapesAll', 'SwedishLeaf', 'WordSynonyms']


def getSets(dataset, ex):
    
    train = ex.loc[(ex['data'] == dataset) & (ex['set'] == 'train')].reset_index().iloc[:,1:]
    test  = ex.loc[(ex['data'] == dataset) & (ex['set'] == 'test')].reset_index().iloc[:,1:]

    X_train = train.iloc[:,:-3]
    y_train = train['label']

    X_test = test.iloc[:,:-3]
    y_test = test['label']
    
    return X_train, y_train, X_test, y_test

def plotShapValues(shapValues, featNames, nbrTopFeats):
    
    
    # stores the shap values of each feature for each instance
    df = pd.DataFrame(shapValues, columns = featNames)
    
    # absolute value of each entry
    df = df.apply(abs)
    
    # calculate the sum of shap values for each feature
    sums = df.sum(numeric_only=True, axis=0).sort_values(ascending=False).to_frame().T
    
    
    endFrame = sums.iloc[:,0:nbrTopFeats].copy()
    name = 'Sum of ' + str(len(sums.loc[0]) - nbrTopFeats) + ' other features'
    endFrame[name] = sums.iloc[:,nbrTopFeats:].sum(numeric_only=True, axis=1).values
    
    ax = sns.barplot(data=endFrame, orient='h', color='darkgreen')
    
    for bars in ax.containers:
        ax.bar_label(bars, color='darkgreen')
    
    ax.set(xlabel='mean (|SHAP value|)')
    plt.show()
    



from sklearn.ensemble import RandomForestClassifier

def computeSHAP_kernelExp(X_train, y_train, X_test):
    
    # Prepares a default instance of the random forest classifier
    model = RandomForestClassifier()
    
    # Fits the model on the data
    model.fit(X_train, y_train)

    sampledata = shap.sample(X_test, 100)
    # Fits the explainer
    explainer = shap.KernelExplainer(model.predict, sampledata)
    
    # Calculates the SHAP values - It takes some time
    shap_values = explainer.shap_values(X_test[:100])

    plotShapValues(shap_values, X_test.columns, 10)


# --------------------------------------
