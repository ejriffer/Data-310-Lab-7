# Data-310-Lab-7

## Question 1: From sklearn import the diabetes data set and assign names for the input variables: If we use RandomForest (random_state=310) max_depth=10 and 1000 trees for ranking the importance of the input features the top three features are (in decreasing order): 

from sklearn.datasets import load_diabetes

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

columns = 'age gender bmi map tc ldl hdl tch ltg glu'.split()  
 
diabetes = load_diabetes()  

df = pd.DataFrame(diabetes.data, columns=columns)  

y = diabetes.target  

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=310, max_depth=10,n_estimators=1000)

df=pd.get_dummies(df)

model.fit(df,y)

features = df.columns

importances = model.feature_importances_

indices = np.argsort(importances)[-9:]  # top 10 features

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()

<img width="383" alt="Screen Shot 2021-05-06 at 1 59 08 PM" src="https://user-images.githubusercontent.com/74326062/117344310-3cd50e80-ae73-11eb-85f4-c284466679a3.png">

## Question 2: For the diabetes dataset you worked on the previous question, apply stepwise regression with add/drop p-values both set to 0.001. The model selected has the following input variables:

import statsmodels.api as sm

def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.001, 
                       threshold_out = 0.001, 
                       verbose=True):
                       
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details """
    
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included
    
result = stepwise_selection(df,y)

output: 

Add  bmi                            with p-value 3.46601e-42

Add  ltg                            with p-value 3.03968e-20

Add  map                            with p-value 3.74192e-05

## Question 3: For the diabetes dataset scale the input features by z-scores and then apply the ElasticNet model with alpha=0.1 and l1_ratio=0.5. If we rank the variables in the decreasing order of the absolute value of the coefficients  the top three variables (in order) are:

def zto(x):

  return (x-np.mean(x))/(np.std(x))
  
xscaled = zto(df)

from sklearn import linear_model as lm

model = lm.ElasticNet(alpha=0.1,l1_ratio = 0.5)

model.fit(xscaled,y)

v = -np.sort(-np.abs(model.coef_))

for i in range(xscaled.shape[1]):

  print(xscaled.columns[np.abs(model.coef_)==v[i]])
  
model.coef_

output: 

Index(['bmi'], dtype='object')

Index(['ltg'], dtype='object')

Index(['map'], dtype='object')

## Question 5: In this problem consider 10-fold cross-validations and random_state=1693 for cross-validations and the decision tree. If you analyze the data with  benign/malign tumors from breast cancer data with two features (radius_mean  and texture_mean) and, according to what you learned about model selection,   you try to determine the best maximum depth (in a range between 1 and 100)  and the best  minimum samples per leaf (in a range between 1 and 25) the  optimal pair of hyper-parameters (such as max depth and min leaf samples) is:

from sklearn.model_selection import KFold

from sklearn import tree

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

dat = load_breast_cancer()

X = pd.DataFrame(data=dat.data, columns=dat.feature_names)

x1 = pd.DataFrame(X['mean radius'])

x2 = pd.DataFrame(X['mean texture'])

x = pd.concat([x1,x2], axis = 1)

x = np.array(x)

y = dat.target

scaler = StandardScaler()

xscaled = scaler.fit_transform(x)

model = tree.DecisionTreeClassifier(random_state=1693) 

params = [{'max_depth':np.arange(1,101),'min_samples_leaf':np.arange(2,26)}]

gs = GridSearchCV(estimator=model,cv=10,scoring='neg_mean_squared_error',param_grid=params)

gs_results = gs.fit(x,y)

print(gs_results.best_params_)

print('The best MSE is: ', np.abs(gs_results.best_score_))

output: 

{'max_depth': 4, 'min_samples_leaf': 23}
The best MSE is:  0.10902255639097744

## Question 6: In this problem consider 10-fold cross-validations and random_state=12345  for cross-validations and the decision tree. If you analyze the data with  benign/malign tumors from breast cancer data with two features (radius_mean and  texture_mean) and, according to what you learned about model selection,  you  try to determine the best maximum depth (in a range between 1 and 100) and the  best  minimum samples per leaf (in a range between 1 and 25) the number of  False Negatives is:

model = tree.DecisionTreeClassifier(random_state=12345)

params = [{'max_depth':np.arange(1,101),'min_samples_leaf':np.arange(2,26)}]

gs = GridSearchCV(estimator=model,cv=10,scoring='neg_mean_squared_error',param_grid=params)

gs_results = gs.fit(xscaled,y)

print(gs_results.best_params_)

print('The best MSE is: ', np.abs(gs_results.best_score_))

output: 

{'max_depth': 4, 'min_samples_leaf': 23}
The best MSE is:  0.10723684210526314

model = tree.DecisionTreeClassifier(max_depth = 4, min_samples_leaf= 23, random_state=12345)  

model.fit(xscaled, y)

y_hat = model.predict(xscaled)

from sklearn.metrics import confusion_matrix as CM

cm = CM(y,y_hat)

spc = ['Begnin','Malignent']

pd.DataFrame(cm, columns=spc, index=spc)

<img width="251" alt="Screen Shot 2021-05-06 at 2 09 22 PM" src="https://user-images.githubusercontent.com/74326062/117345540-aa356f00-ae74-11eb-9007-84767d72c3bc.png">

## Question 7: In this problem consider 10-fold cross-validations and random_state=1693  for cross-validations and the decision tree. If you analyze the data with  benign/malign tumors from breast cancer data set with two features (radius_mean and texture_mean) and, according to what you learned about model selection, you try to determine the best maximum depth (in a range between 1 and 100) and  the best  minimum samples per leaf (in a range between 1 and 25) the  accuracy is about:

from sklearn.metrics import accuracy_score

model = tree.DecisionTreeClassifier(max_depth = 4, min_samples_split= 23, random_state=1693)  

model.fit(xscaled, y)

predicted_classes = model.predict(xscaled)

accuracy = accuracy_score(y,predicted_classes)

print(accuracy)

output: 0.9156414762741653

## Question 12: In this problem the input features will be scaled by the z-scores  and consider a use a random_state=1234. If you analyze the data with benign/malign tumors from breast cancer data, consider a decision tree with max_depth=10, min_samples_leaf=20 and fit on 9 principal components the number of true positives is:

def zto(x):

  return (x-np.mean(x))/(np.std(x))
  
xscaled = zto(X)

from sklearn.decomposition import PCA

pca = PCA(n_components=9, random_state = 1234)

principalComponents = pca.fit_transform(xscaled)

principalDf = pd.DataFrame(data = principalComponents,
             columns = ['1','2','3','4','5','6','7','8','9'])
             
Y = pd.DataFrame(y)

finalDf = pd.concat([principalDf, Y], axis = 1)

model = tree.DecisionTreeClassifier(max_depth = 10, min_samples_leaf= 20, random_state=1234) 

model.fit(principalDf,y)

y_hat = model.predict(principalDf)

cm = CM(y,y_hat)

spc = ['Begnin','Malignent']

pd.DataFrame(cm, columns=spc, index=spc)

<img width="251" alt="Screen Shot 2021-05-06 at 2 11 49 PM" src="https://user-images.githubusercontent.com/74326062/117345836-01d3da80-ae75-11eb-895e-26ba7b1f454a.png">
