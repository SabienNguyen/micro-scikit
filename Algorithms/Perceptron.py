import numpy as np

class Perceptron:
    
    def __init__(self, input_dim, learning_rate=0.01, n_epochs=1000):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = np.zeros(input_dim)
        self.bias = 0.0

    def activation(self, z):
        # z is expected to be a scalar
        print(z)
        return 1 if z.any() > 0 else 0

    def forward(self, x):
        # Ensure x is a 1D array for dot product
        z = np.dot(x, self.weights) + self.bias
        return self.activation(z)

    def train(self, X, y):
        for epoch in range(self.n_epochs):
            for xi, yi in zip(X, y):
                print(xi)
                xi = np.asarray(xi).flatten()
                y_pred = self.forward(xi)
                error = yi - y_pred
                self.weights += self.learning_rate * error * xi
                self.bias += self.learning_rate * error

    def predict(self, X):
        return self.forward(X)
