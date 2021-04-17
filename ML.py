import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

def linear_regression(X, y, test_ratio):

    # standardize features with z-score
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
    
    clf = linear_model.Lasso(alpha=0.1).fit(X_train, y_train)
    pred = clf.predict(X_test)

    print('Coefficients: ', clf.coef_)
    print('Predicted age: ', pred)
    print('Real age: ', y_test)