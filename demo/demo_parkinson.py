
import time

import pandas as pd
pd.set_option("display.max_columns", None)

import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.multioutput import MultiOutputRegressor
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from demo_tools import calc_features, get_data, compute_metrics

def training_linear_models(X_train, Y_train, X_test, Y_test, times, CV_times, r2, MSE):
    # We initialize the feature transformation steps (e.g., scaling, dimensionality reduction) without hyperparameters
    scaler = StandardScaler()
    pca = PCA()
    n_components = [3, 5, 7, 8, 10, 12, 13]
    start_time = time.time()
    
    # Linear Regression
    clf = MultiOutputRegressor(LinearRegression())
    clf.fit(X_train, Y_train)
    train_predict, test_predict, train_accuracy, test_accuracy, train_MSE, test_MSE = compute_metrics(clf, X_train, Y_train, X_test, Y_test)
    times.append(time.time() - start_time)
    print(" | Linear Regression | Total fit and evaluation time: %s seconds" % (time.time() - start_time))
    CV_times.append(0)
    r2.append(test_accuracy)
    MSE.append(test_MSE)
    
    # Polynomial Regression
    pf2 = PolynomialFeatures(degree=2)
    x_train2 = pf2.fit_transform(X_train)
    x_test2 = pf2.fit_transform(X_test)
    start_time = time.time()
    clf = lr = MultiOutputRegressor(LinearRegression())
    lr.fit(x_train2, Y_train)
    result = compute_metrics(clf, x_train2, Y_train, x_test2, Y_test)
    train_predict, test_predict, train_accuracy, test_accuracy, train_MSE, test_MSE = result
    times.append(time.time() - start_time)
    print(" | Polynomial Regression | Total fit and predict time: %s seconds" % (time.time() - start_time))
    CV_times.append(0)
    r2.append(test_accuracy)
    MSE.append(test_MSE)
    
    # Elastic Net
    start_time = time.time()
    en = ElasticNet()
    en.fit(X_train, Y_train)
    compute_metrics(en, X_train, Y_train, X_test, Y_test)
    times.append(time.time() - start_time)
    print(" | Elastic Net | Initialization. Total fit and predict time: %s seconds" % (time.time() - start_time))
    # In Elastic Net, there are several hyperparameters that need to be tuned. 
    # For this reason, we perform a grid search with cross-validation to find the optimal values.
    # Indeed, we observe that after this tuning process, the algorithm's ability to predict and generalize improves significantly.
    l1 = np.arange(0.1,1,0.2)
    alphas = np.arange(0,2,0.1)
    pipe_en = Pipeline(steps=[('scaler', scaler), ('pca', pca), ('ElasticNet', en)], memory = 'tmp')
    estimator_en = GridSearchCV(pipe_en, dict( pca__n_components=n_components, ElasticNet__alpha=alphas, 
                                                ElasticNet__l1_ratio=l1), cv=5, scoring ='r2', n_jobs=-1)
    start_time = time.time()
    estimator_en.fit(X_train, Y_train)
    result = compute_metrics(estimator_en, X_train, Y_train, X_test, Y_test)
    train_predict, test_predict, train_accuracy, test_accuracy, train_MSE, test_MSE = result
    CV_times.append(time.time() - start_time)
    print(" | Elastic Net | GridSearch Cross-Validation. Total fit and predict time: %s seconds" % (time.time() - start_time))
    r2.append(test_accuracy)
    MSE.append(test_MSE)
    
    # Lasso Regression
    start_time = time.time()
    lasso = linear_model.MultiTaskLasso()
    lasso.fit(X_train, Y_train)
    compute_metrics(lasso, X_train, Y_train, X_test, Y_test)
    times.append(time.time() - start_time)
    print(" | Lasso Regression | Initialization. Total fit and predict time: %s seconds" % (time.time() - start_time))
    # In Lasso Regression, there are several hyperparameters that need to be tuned. 
    # For this reason, we perform a grid search with cross-validation to find the optimal values.
    # Indeed, we observe that after this tuning process, the algorithm's ability to predict and generalize improves significantly.
    alphas = np.arange(0.1,2,0.1)
    pipe_lasso = Pipeline(steps=[('scaler', scaler), ('pca', pca), ('lasso', lasso)], memory = 'tmp')
    estimator_lasso = GridSearchCV(pipe_lasso, dict( pca__n_components=n_components, lasso__alpha=alphas ), cv=5, scoring ='r2', n_jobs=-1)
    start_time = time.time()
    estimator_lasso.fit(X_train, Y_train)
    result = compute_metrics(estimator_lasso, X_train, Y_train, X_test, Y_test)
    train_predict, test_predict, train_accuracy, test_accuracy, train_MSE, test_MSE = result
    CV_times.append(time.time() - start_time)
    print(" | Lasso Regression | GridSearch Cross-Validation. Total fit and predict time: %s seconds" % (time.time() - start_time))

    r2.append(test_accuracy)
    MSE.append(test_MSE)
    
    return times, CV_times, r2, MSE
    
def training_nonlinear_models(X_train, Y_train, X_test, Y_test, times, CV_times, r2, MSE):
    scaler = StandardScaler()
    
    # Decission Tree Regressor
    start_time = time.time()
    tree = MultiOutputRegressor(DecisionTreeRegressor(random_state=0))
    tree.fit(X_train, Y_train)
    compute_metrics(tree, X_train, Y_train, X_test, Y_test)
    times.append(time.time() - start_time)
    print(" | Decission Tree | Initialization. Total fit and predict time: %s seconds" % (time.time() - start_time))
    alphas = np.arange(0.0,2.0,0.2)
    pipe_tree = Pipeline(steps=[('scaler', scaler), ('tree', tree)], memory = 'tmp')
    treeCV = GridSearchCV(pipe_tree, dict(tree__estimator__ccp_alpha=alphas ), cv=5, scoring ='r2', n_jobs=-1)
    start_time = time.time()
    treeCV.fit(X_train, Y_train)
    result = compute_metrics(treeCV, X_train, Y_train, X_test, Y_test)
    train_predict, test_predict, train_accuracy, test_accuracy, train_MSE, test_MSE = result
    CV_times.append(time.time() - start_time)
    print(" | Decission Tree | GridSearch Cross-Validation. Total fit and predict time: %s seconds" % (time.time() - start_time))
    r2.append(test_accuracy)
    MSE.append(test_MSE)
    
    # k-Nearest Neighbors
    neigh = MultiOutputRegressor(KNeighborsRegressor(n_neighbors=3))
    start_time = time.time()
    neigh.fit(X_train, Y_train)
    compute_metrics(neigh, X_train, Y_train, X_test, Y_test)
    times.append(time.time() - start_time)
    print(" | k-Nearest Neighbors | Initialization. Total fit and predict time: %s seconds" % (time.time() - start_time))
    algos = ['auto', 'ball_tree', 'kd_tree', 'brute']
    num = [1,2,3,4,5,6]
    pipe_neigh = Pipeline(steps=[('scaler', scaler), ('neigh', neigh)], memory = 'tmp')
    neighCV = GridSearchCV(pipe_neigh, dict( neigh__algorithm =algos, neigh__n_neighbors=num ), cv=5, scoring ='r2', n_jobs=-1)
    start_time = time.time()
    treeCV.fit(X_train, Y_train)
    result = compute_metrics(treeCV, X_train, Y_train, X_test, Y_test)
    train_predict, test_predict, train_accuracy, test_accuracy, train_MSE, test_MSE = result
    CV_times.append(time.time() - start_time)
    print(" | k-Nearest Neighbors | GridSearch Cross-Validation. Total fit and predict time: %s seconds" % (time.time() - start_time))
    r2.append(test_accuracy)
    MSE.append(test_MSE)
    
    # Random Forest
    rd_for = MultiOutputRegressor(RandomForestRegressor(random_state=0))
    start_time = time.time()
    rd_for.fit(X_train, Y_train)
    compute_metrics(rd_for, X_train, Y_train, X_test, Y_test)
    times.append(time.time() - start_time)
    print(" | Random Forest | Initialization. Total fit and predict time: %s seconds" % (time.time() - start_time))
    alphas = np.arange(0,1,0.2)
    pipe_rdfor = Pipeline(steps=[('scaler', scaler), ('rd_for', rd_for)], memory = 'tmp')
    rdforCV = GridSearchCV(pipe_rdfor, dict( rd_for__estimator__ccp_alpha = alphas ), cv=5, scoring ='r2', n_jobs=-1)
    start_time = time.time()
    rdforCV.fit(X_train, Y_train)
    train_predict, test_predict, train_accuracy, test_accuracy, train_MSE, test_MSE = result
    result = compute_metrics(rdforCV, X_train, Y_train, X_test, Y_test)
    CV_times.append(time.time() - start_time)
    print(" | Random Forest | GridSearch Cross-Validation. Total fit and predict time: %s seconds" % (time.time() - start_time))
    r2.append(test_accuracy)
    MSE.append(test_MSE)
    
    # Gradient Boosting Regressor
    GBR = MultiOutputRegressor(GradientBoostingRegressor(random_state=0))
    start_time = time.time()
    GBR.fit(X_train, Y_train)
    compute_metrics(GBR, X_train, Y_train, X_test, Y_test)
    times.append(time.time() - start_time)
    print(" | Gradient Boosting Regressor | Initialization. Total fit and predict time: %s seconds" % (time.time() - start_time))
    alphas = np.arange(0.0,1.0,0.1)
    pipe_gbr = Pipeline(steps=[('scaler', scaler), ('GBR', GBR)], memory = 'tmp')
    gbrCV = GridSearchCV(pipe_gbr, dict( GBR__estimator__alpha = alphas ), cv=5, scoring ='r2', n_jobs=-1)
    start_time = time.time()
    gbrCV.fit(X_train, Y_train)
    result = compute_metrics(gbrCV, X_train, Y_train, X_test, Y_test)
    train_predict, test_predict, train_accuracy, test_accuracy, train_MSE, test_MSE = result 
    CV_times.append(time.time() - start_time)
    print(" | Gradient Boosting Regressor | GridSearch Cross-Validation. Total fit and predict time: %s seconds" % (time.time() - start_time))
    r2.append(test_accuracy)
    MSE.append(test_MSE)
    print(f" | Gradient Boosting Regressor | KEYS: {GBR.get_params().keys()}")

def draw_diagrams(times, CV_times, r2, MSE):
    names = ['Linear', 'Polynomial', 'Elastic-Net', 'Lasso', 'Tree', 'kNN', 'Forest', 'BGR']
    
    plt.figure(figsize=(10, 6))
    plot_data = {'names': names, 'times': times}
    sns.set_theme(style="whitegrid")
    tips = sns.load_dataset("tips")
    ax = sns.barplot(x="names", y="times", data=plot_data).set_title("Algorithm Times Barplot",fontsize=16)
    plt.tight_layout()
    plt.savefig("../images/algorithm_times.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plot_data = {'names': names, 'tuning times': CV_times}
    sns.set_theme(style="whitegrid")
    tips = sns.load_dataset("tips")
    ax = sns.barplot(x="names", y="tuning times", data=plot_data).set_title("Algorithm Tuning Times Barplot",fontsize=16)
    plt.tight_layout()
    plt.savefig("../images/tuning_times.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plot_data = {'names': names, 'R2 metric': r2}
    sns.set_theme(style="whitegrid")
    tips = sns.load_dataset("tips")
    ax = sns.barplot(x="names", y="R2 metric", data=plot_data).set(title='R2 metric bar plot', ylim=(0.9, 1.0))
    plt.tight_layout()
    plt.savefig("../images/r2_metric.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plot_data = {'names': names, 'MSE metric': MSE}
    sns.set_theme(style="whitegrid")
    tips = sns.load_dataset("tips")
    ax = sns.barplot(x="names", y="MSE metric", data=plot_data).set_title("MSE metric Barplot",fontsize=16)
    plt.tight_layout()
    plt.savefig("../images/mse_metric.png")
    plt.close()

def training_models(X_train, Y_train, X_test, Y_test):
    times = []
    CV_times = []
    r2 = []
    MSE = []
    training_linear_models(X_train, Y_train, X_test, Y_test, times, CV_times, r2, MSE)
    training_nonlinear_models(X_train, Y_train, X_test, Y_test, times, CV_times, r2, MSE)
    draw_diagrams(times, CV_times, r2, MSE)

if __name__ == "__main__":
    X, Y, dataset = get_data()
    training_models(*calc_features(X, Y, dataset))



