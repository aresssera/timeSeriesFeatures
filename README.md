# timeSeriesFeatures

Compares the feature sets of the extractors _catch22_, _Kats_, _tsfeatures_, _TSFEL_ and _tsfresh_, with respect to their computation time, internal structure and relationships between the different sets.
Furthermore, it evaluates how selected classification and regression models are affected by these features, identifying which features have the most significant impact on the model.





## Prerequisites

Have the following libraries installed:

- seaborn
- numpy
- pandas
- feather
- sklearn
- sktime
- shap
- pycatch22
- kats
- tsfeatures
- tsfel
- tsfresh


## Feature Extractor Comparison

The analysis introduced in https://github.com/hendersontrent/feature-set-comp was used as a guideline.

### Execution
All tests can be run immidiately.


If you want to use your own data, you can calculate the features using _calcFeat_. 
_compTime_ can be used directly. For all others, _correlationMatrices_ must be run first.

### Output

The outputs are stored in the correspongind folder _output_


## Feature Impact

Six commonly used models were tested for each task.
In both cases, the features of _TSFEL_ and _tsfresh_ were investigated in combination with _Random Forest_ models.

### Execution

All tests can be run. If 

### Output

The output plots are not stored in a directory.


```
sdofhkdfh
```
