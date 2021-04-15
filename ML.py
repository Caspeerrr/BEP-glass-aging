import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def linear_regression(X, y, test_ratio):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)

    reg = LinearRegression().fit(X_train, y_train)
    pred = reg.predict(X_test)

    print('Predicted age: ', pred)
    print('Real age: ', y_test)