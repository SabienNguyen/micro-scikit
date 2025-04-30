# Steps to Building out a logisitic Regression Model
# Logistic function in this case is a sigmoid
# take the linear combination of weights and inputs
# training function needed
# Need to create logistic regression class

import numpy as np

class Logistic_Regression:
    
    def __init__(self):
        return
    
    def _sigmoid(self, z):
        return 1 / (1 + np.e ** (-1 * z))
    
    def train(self, alpha):
        # steps
        # figure out stopping point (when we hit gradient)
        # initialize weights and bias
        # training loop
        # need to define a loss function to measure performance
        # take linear combination of training data and weights
        # predict on all inputs
        # find loss
        # adjust weights accordingly
        pass
    
    def compute_linear_combination(self, weights, input, bias):
        return np.dot(weights, input) + bias
    
    def predict(self, input):
        return 1 if self.sigmoid(self.compute_linear_combination(input)) > 0.5 else 0
    