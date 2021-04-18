import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score
from sklearn import linear_model

def linear_regression(X, y, test_ratio):

    # standardize features with z-score
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
    
    clf = linear_model.Lasso(alpha=0.1).fit(X_train, y_train)
    pred = clf.predict(X_test).astype(int)

    print('Coefficients: ', clf.coef_)
    print('Predicted age: ', pred)
    print('Real age: ', y_test)

def logistic_regression(X, y, test_ratio):

    # standardize features with z-score
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

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
