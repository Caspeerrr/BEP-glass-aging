import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

def linear_regression(X, y, test_ratio):

    # standardize features with z-score
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train1, X_test1, y_train1, y_test1 = train_test_split(X[:30], y[:30], test_size=test_ratio, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X[70:], y[70:], test_size=test_ratio, random_state=42)

    X_train, X_test, y_train, y_test = np.concatenate((X_train1, X_train2)), np.concatenate((X_test1, X_test2)), np.concatenate((y_train1, y_train2)), np.concatenate((y_test1, y_test2))

    clf = linear_model.Lasso(alpha=0.1).fit(X_train, y_train)
    pred = clf.predict(X_test).astype(int)

    print('Coefficients: ', clf.coef_)
    print('Predicted age: ', pred)
    print('Real age: ', y_test)