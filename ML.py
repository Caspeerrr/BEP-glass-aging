import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import precision_score, recall_score
from sklearn import linear_model


"""
Machine learning methods available:
    - Linear regression(X, y, test_ratio)
    - Logistic regression(X, y, test_ratio)
"""


def linear_regression(X, y, test_ratio):

    # standardize features with z-score
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # second degree polynomial
    poly = PolynomialFeatures(2)
    X = poly.fit_transform(X)
    print(poly.get_feature_names())
    # print(poly.get_params())

    # divide in training and test set and predict
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
    clf = linear_model.Lasso(alpha=0.1).fit(X_train, y_train)
    pred = clf.predict(X_test).astype(int)

    print('Coefficients: ', clf.coef_)
    print('Parameters:', clf.get_params())
    print('Predicted age: ', pred)
    print('Real age: ', y_test)

    R2_test = clf.score(X_test, y_test)
    R2_train = clf.score(X_train, y_train)
    print('R2 training set:', R2_train)
    print('R2 test set:', R2_test)


def logistic_regression(X, y, test_ratio):

    # standardize features with z-score
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # divide in training and test set and predict
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
    clf = LogisticRegression(penalty='l2').fit(X_train, y_train)
    pred = clf.predict(X_test)
    prob = clf.predict_proba(X_test)

    accuracy = clf.score(X_test, y_test)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)

    print('Predictions:', pred)
    print('with probabilities:', prob)
    print('accuracy logistic regression:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
