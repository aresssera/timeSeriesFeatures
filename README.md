# timeSeriesFeatures

Compares the feature sets of the extractors _catch22_, _Kats_, _tsfeatures_, _TSFEL_ and _tsfresh_, with respect to their computation time, internal structure and relationships between the different sets.
Furthermore, it evaluates how selected classification and regression models are affected by these features, identifying which features have the most significant impact on the model.





## Prerequisites

Have the following libraries installed:

- Jupyter notebook environment
- Python, including the following librariers:
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
All notebooks can be run immediately. The steps are described in the notebooks.


If you want to use your own data, you can calculate the features using `calcFeat`. 
After that, the computation time comparison can be directly done using `compTime`. For all tests, `correlationMatrices` must be run first.

### Output

The outputs are stored in the correspongind folder `/outputs`.


## Feature Impact

For each task, six commonly used models were tested in combination with the mentioned feature sets.
In both cases, the features of _TSFEL_ and _tsfresh_ were further investigated in combination with the corresponding _Random Forest_ model.

<table>
<tr><th>Table 1 Heading 1 </th><th>Table 1 Heading 2</th></tr>
<tr><td>

|Table 1| Middle | Table 2|
|--|--|--|
|a| not b|and c |

</td><td>

|b|1|2|3| 
|--|--|--|--|
|a|s|d|f|

</td></tr> </table>

### Execution

All notebooks can be run immediately. Comments that are (hopefully) useful are directly included in the files.

### Output

The output plots are not stored in a directory. 

