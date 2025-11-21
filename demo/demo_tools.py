
import os

import statsmodels.api as sm
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import pyplot

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoCV

PARKINSONS_TELEMONITORING_DATA_PATH = "../data/parkinsons_telemonitoring_colab.csv"

def compute_metrics(model, x_tr, y_tr, x_ts, y_ts, model_type='sklearn', y_scaler=None):
    if model_type == 'sklearn':
        train_predict = model.predict(x_tr)
        test_predict = model.predict(x_ts)
    elif model_type in ['keras', 'quantum']:
        train_predict = model.predict(x_tr, verbose=0)
        test_predict = model.predict(x_ts, verbose=0)
    else:
        raise ValueError("Unsupported model_type")

    # Inverse-transform predictions if a scaler is provided
    if y_scaler is not None:
        train_predict = y_scaler.inverse_transform(train_predict)
        test_predict = y_scaler.inverse_transform(test_predict)

    train_accuracy = r2_score(y_tr, train_predict)
    test_accuracy = r2_score(y_ts, test_predict)
    train_MSE = mean_squared_error(y_tr, train_predict)
    test_MSE = mean_squared_error(y_ts, test_predict)

    print(f'Accuracy on training dataset: {train_accuracy:.2%}')
    print(f'Accuracy on test dataset: {test_accuracy:.2%}')
    print(f'Mean Squared Error on training samples: {train_MSE:.4f}')
    print(f'Mean Squared Error on test samples: {test_MSE:.4f}')

    return train_predict, test_predict, train_accuracy, test_accuracy, train_MSE, test_MSE

def select_features(X_train, y_train, X_test):
	fs = SelectKBest(score_func=mutual_info_regression, k='all')
	fs.fit(X_train, y_train)
	X_train_fs = fs.transform(X_train)
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

def select_features_f_score(X_train, Y_train, X_test):
	fs = SelectKBest(score_func=f_regression, k='all')
	fs.fit(X_train, Y_train)
	X_train_fs = fs.transform(X_train)
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

"""
Correlation
==========
Correlation is a measure that indicates whether two random variables depend on each other. 
The most common correlation metric is Pearson correlation, which assumes the random variables 
follow a Gaussian (normal) distribution and measures their linear dependency.

Linear correlation values typically range from -1 to 1, with 0 indicating no correlation. 
For feature selection, we are particularly interested in positive correlations. 
The higher the correlation between a feature and the target variable, 
the more likely it is that this feature will help us build a good predictive model.

One way to assess how correlated our features are with the target variable is by using the F-score. 
The F-score is a statistical measure of correlation; the higher the F-score of a feature, 
the more likely we are to retain it in our final model.

In our case, since we aim to predict two target variables (motor_UPDRS and total_UPDRS), 
we should examine the correlations with each of them separately. As a starting point, 
we can generate a bar plot of the F-scores for the first target variable, motor_UPDRS.
"""
def feat_sel_f_score(X_train, Y_train, X_test):
    pyplot.clf()
    X_train_fs, X_test_fs, fs = select_features_f_score(X_train, Y_train[:,0], X_test) # motor_UPDRS
    for i in range(len(fs.scores_)):
        print('Feature %d: %f' % (i, fs.scores_[i]))
    pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
    pyplot.savefig("../images/feature_selection_f_score_motor_UPDRS.png")

    pyplot.clf()
    X_train_fs, X_test_fs, fs = select_features_f_score(X_train, Y_train[:,1], X_test) # total_UPDRS
    for i in range(len(fs.scores_)):
        print('Feature %d: %f' % (i, fs.scores_[i]))
    pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
    pyplot.savefig("../images/feature_selection_f_score_total_UPDRS.png")

"""
Mutual Information Feature Selection
=========================================
Mutual information is a powerful technique for feature selection, 
especially when dealing with non-linear relationships between features and target variables.

This is a measure derived from information theory. 
Mutual information is calculated between two variables and quantifies the reduction in uncertainty 
about one random variable given knowledge of the other.

It can be used in a similar way to the F-score. 
Therefore, we can assume that the top k features are those with the highest mutual information scores.
"""
def feat_sel_mutual(X_train, Y_train, X_test):
    pyplot.clf()
    X_train_fs, X_test_fs, fs = select_features(X_train, Y_train[:,0], X_test) # motor_UPDRS
    for i in range(len(fs.scores_)):
        print('Feature %d: %f' % (i, fs.scores_[i]))
    pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
    pyplot.savefig("../images/feature_selection_mutual_motor_UPDRS.png")
    
    pyplot.clf()
    X_train_fs, X_test_fs, fs = select_features(X_train, Y_train[:,1], X_test) # total_UPDRS
    for i in range(len(fs.scores_)):
        print('Feature %d: %f' % (i, fs.scores_[i]))
    pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
    pyplot.savefig("../images/feature_selection_mutual_total_UPDRS.png")

"""
Ηeatmap
==========
On one hand, we want the features to be correlated with the dependent variables y, so they carry useful predictive information. 
On the other hand, for the assumptions of a linear model to hold, we want the features to be as uncorrelated with each other as possible.

Features are considered uncorrelated when their pairwise correlation values are close to zero. 
If they have high absolute correlations (whether positive or negative), this generally indicates a problem, 
such as multicollinearity, which can distort model interpretation and reduce performance. 
Therefore, we aim to retain features that have low inter-feature correlation.

One way to visualize these relationships is by using a heatmap of the correlation matrix.
"""
def feat_sel_heatmap(dataset):
    plt.clf()
    plt.figure(figsize=(12,10))
    cor = dataset.corr()
    sns.heatmap(cor, annot=False, cmap=plt.cm.Reds)
    plt.savefig("../images/heatmap.png")

    """
    Therefore, we can set a threshold and identify which of our random variables (features) are most positively correlated 
    with the target variables y we want to predict.
    In this case, we use a threshold of 0.1. This is one way to perform feature selection: 
    by keeping only the features whose correlation with the target exceeds this value. 
    However, this method only shows how correlated the features are with y — 
    it does not tell us how correlated the features are with each other. 
    That’s why, in practice, we should combine this method with inter-feature correlation analysis (e.g., via a heatmap) 
    to avoid multicollinearity.
    """
    #Correlation with output variable
    cor_target = abs(cor['motor_UPDRS'])
    #Selecting highly correlated features
    relevant_features = cor_target[cor_target>0.1]
    print(f"Highly correlated features with motor_UPDRS: {os.linesep}{relevant_features }")
    
    #Correlation with output variable
    cor_target = abs(cor['total_UPDRS'])
    #Selecting highly correlated features
    relevant_features = cor_target[cor_target>0.1]
    print(f"Highly correlated features with total_UPDRS: {os.linesep}{relevant_features }")


def feat_sel_backward_elimination(X_train, Y_train):
    #Adding constant column of ones, mandatory for sm.OLS model
    X_1 = sm.add_constant(X_train)
    #Fitting sm.OLS model
    model = sm.OLS(Y_train[:,0],X_1).fit()
    print("sm.OLS model's pvalues: ", model.pvalues)
    
    #Backward Elimination
    cols = list(pd.DataFrame(X_train).columns)
    pmax = 1
    while (len(cols)>0):
        p= []
        X_1 = pd.DataFrame(X_train)[cols]
        X_1 = sm.add_constant(X_1)
        model = sm.OLS(pd.DataFrame(Y_train[:,0]),X_1).fit()
        p = pd.Series(model.pvalues.values[1:],index = cols)
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if(pmax>0.05):
            cols.remove(feature_with_p_max)
        else:
            break
    selected_features_BE = cols
    print("Backward Elimination: ", selected_features_BE)

"""
The Recursive Feature Elimination (RFE) method takes two main inputs:
- The number of features we want to retain in the final model.
- A regressor or classifier that will be used to evaluate performance (e.g., via accuracy score, R², etc.).

As output, RFE provides:
- A ranking of all features based on their importance.
- A boolean mask indicating which features were selected.

This approach is particularly useful when you want to reduce dimensionality while directly optimizing for predictive performance.
"""
def feat_sel_recursive_elimination(X_train, Y_train):
    model = LinearRegression()
    #Initializing RFE model
    rfe = RFE(estimator=model, n_features_to_select=7)
    #Transforming data using RFE
    X_rfe = rfe.fit_transform(X_train, Y_train[:,0])
    #Fitting the data to model
    model.fit(X_rfe, Y_train[:,0])
    print(rfe.support_)
    print(rfe.ranking_)

"""
Embedded methods are techniques that select features during the model training process itself, 
by carefully retaining those that contribute the most at each iteration. 
One effective approach is to use regularization, which applies a penalty to less important features through a coefficient-based threshold.

Specifically, we use Lasso regularization (L1 regularization). In Lasso:
- Features that are not useful for the model are assigned a coefficient of zero.
- We can then eliminate all features with zero coefficients, as they do not contribute to the model’s predictions.
- This provides a built-in mechanism for feature selection, making Lasso both a predictive model and 
    a dimensionality reduction technique in one.
"""
def feat_sel_lasso(X_train, Y_train):
    plt.clf()
    reg = LassoCV()
    reg.fit(X_train, Y_train[:,0])
    print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
    print("Best score using built-in LassoCV: %f" % reg.score(X_train, Y_train[:,0]))
    coef = pd.Series(reg.coef_, index = pd.DataFrame(X_train).columns)
    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
    print(f"Lasso coefficients: {os.linesep}{coef}")
    imp_coef = coef.sort_values()
    matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
    imp_coef.plot(kind = "barh")
    plt.title("Feature importance using Lasso Model")
    plt.savefig("../images/lasso.png")

def calc_features(X, Y, dataset):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    print('Train', X_train.shape, Y_train.shape)
    print('Test', X_test.shape, Y_test.shape)

    feat_sel_f_score(X_train, Y_train, X_test)
    feat_sel_mutual(X_train, Y_train, X_test)
    feat_sel_heatmap(dataset)
    feat_sel_backward_elimination(X_train, Y_train)
    feat_sel_recursive_elimination(X_train, Y_train)
    feat_sel_lasso(X_train, Y_train)
    
    X_train = pd.DataFrame(X_train)
    X_train = X_train[[0, 1, 2, 3, 4, 6, 9, 11, 12, 15, 16, 17, 18]]
    X_train = X_train.values

    X_test = pd.DataFrame(X_test)
    X_test = X_test[[0, 1, 2, 3, 4, 6, 9, 11, 12, 15, 16, 17, 18]]
    X_test = X_test.values
    
    return X_train, Y_train, X_test, Y_test

def get_data():
    print("Data source path: ", PARKINSONS_TELEMONITORING_DATA_PATH)
    dataset = pd.read_csv(PARKINSONS_TELEMONITORING_DATA_PATH)
    print(dataset[0:30])
    print("Columns Names: ",dataset.columns)
    print("Columns Count: ",len(dataset.columns))
    print(dataset.shape)
    dataset = dataset.drop(["subject#"], axis=1)
    
    dataset.isnull().sum()
    array = dataset.values
    X1 = array[:,0:4]
    X2 = array[:,6:]
    X = np.hstack((X1,X2)) # Feature matrix (input data)
    print("Input shape: ", X.shape)
    Y = array[:,4:6] # Label matrix (output data) for motor_UPDRS (4) and total_UPDRS (5).
    print("Ouput shape: ", Y.shape)
    return X, Y, dataset
