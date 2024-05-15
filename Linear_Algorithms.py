#TODO create logistic regression function
import numpy as np

def SimpleLinearRegression(X, y) :
    m = 0 
    b = 0
    
    X_bar = np.mean(X)
    y_bar = np.mean(y)
    
    # Calculate slope and intercept using Least Squares
    m = np.sum((X - X_bar) * (y - y_bar)) / np.sum((X - X_bar) ** 2)
    b = y_bar - m * X_bar
    
    return (m, b)

def MutipleLinearRegression(X, y) :
    m = 0 
    b = 0
    
    X_bar = []
    for i in range(len(X)):
        X_bar.append(np.mean(X[i]))
    y_bar = np.mean(y)
    
    # Calculate slope and intercept using Least Squares
    m = []
    b = y_bar
    for i in range(len(X)):
        m.append(np.sum((X[i] - X_bar[i]) * (y - y_bar)) / np.sum((X[i] - X_bar[i]) ** 2))
    for i in range(len(X)):
        b -= m[i] * X_bar[i]
        
    
    return (m, b)