#TODO create linear regression function
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