# Steps to Building out a logisitic Regression Model
# Logistic function in this case is a sigmoid
# take the linear combination of weights and inputs
# training function needed
# Need to create logistic regression class

import numpy as np

class Logistic_Regression:
    
    def __init__(self):
        self.weights = 0
        self.bias = 0
        return
    
    def _sigmoid(self, z):
        return 1 / (1 + np.e ** (-1 * z))
    
    def _log_loss(self, y, y_hat):
        return -1 * (y * np.log(y_hat) + (1-y) * np.log(1 - y_hat))
    
    def _gradient(self):
        return
    
    def train(self, X, y, alpha):
        # steps
        # figure out stopping point (when we hit gradient)
        # initialize weights and bias
        # training loop
        prev_loss = 0
        while True:
            self.weights = np.zeros(X.shape[1])
            loss = 0
            for i, (_, row) in enumerate(X.iterrows()):
                y_hat = self.predict(input)
                loss += self._log_loss(y[i], y_hat)
            loss /= X.shape[0]
            if np.abs(loss - prev_loss) < 0.0001 :
                break
            prev_loss = loss
        # need to define a loss function to measure performance using log loss
        # take linear combination of training data and weights
        # predict on all inputs
        # find loss
        # adjust weights accordingly
        return
    
    def _compute_linear_combination(self, input):
        return np.dot(self.weights, input) + self.bias
    
    def predict(self, input):
        return 1 if self.sigmoid(self._compute_linear_combination(input)) > 0.5 else 0
    