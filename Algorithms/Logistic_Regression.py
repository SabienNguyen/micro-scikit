# Steps to Building out a logisitic Regression Model
# Logistic function in this case is a sigmoid
# take the linear combination of weights and inputs
# training function needed
# Need to create logistic regression class

import numpy as np
import pandas as pd

class Logistic_Regression:
    
    def __init__(self):
        self.weights = None
        self.bias = 0.0
        return
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _log_loss(self, y, y_hat):
        y_hat = np.clip(y_hat, 1e-9, 1 - 1e-9)
        return -1 * (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    
    def train(self, X, y, alpha=0.1, max_iter=10000, tol=1e-4):
        if isinstance(X, pd.DataFrame):
            X = X.values
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        prev_loss = float('inf')
        
        for _ in range(max_iter):
            linear_model = self._compute_linear_combination(X)
            y_hat = self._sigmoid(linear_model)
            
            #gradients
            error = y_hat - y
            dw = np.dot(X.T, error) / n_samples
            db = np.mean(error)
            
            #update
            self.weights -= alpha * dw
            self.bias -= alpha * db
            
            #compute loss
            loss = np.mean(self._log_loss(y, y_hat))
            if abs(prev_loss - loss) < tol: 
                break
            prev_loss = loss
        # need to define a loss function to measure performance using log loss
        # take linear combination of training data and weights
        # predict on all inputs
        # find loss
        # adjust weights accordingly
        return
    
    def _compute_linear_combination(self, input):
        return np.dot(input, self.weights) + self.bias
    
    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)
    
    def predict(self, input):
        return (self.predict_proba(input) > 0.5).astype(int)
    