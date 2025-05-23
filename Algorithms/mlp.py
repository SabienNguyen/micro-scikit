import numpy as np

class MLP:
    def __init__(self, layers) -> None:
        self.layers = layers    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad, learning_rate):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)
    
    def predict(self, x):
        logits = self.forward(x)
        return np.argmax(logits, axis=1)
        

class Dense:
    def __init__(self, input_dim, output_dim) -> None:
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros((1, output_dim))
    
    def forward(self, x):
        self.input = x
        self.output = x @ self.weights + self.bias
        return self.output
    
    def backward(self, grad_output, learning_rate):
        grad_input = grad_output @ self.weights.T
        grad_weights = self.input.T @ grad_output
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        
        return grad_input
    
    
class ReLU:
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output, learning_rate):
        return grad_output * (self.input > 0)
    

class Softmax:
    def forward(self, x):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exp / np.sum(exp, axis=1, keepdims=True)
        return self.output

    def backward(self, grad_output, learning_rate):
        # This will be combined with cross-entropy, so we won't use this directly
        return grad_output

def cross_entropy_loss(logits, y_true): 
    m = y_true.shape[0]
    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    correct_logprobs = -np.log(probs[range(m), y_true])
    loss = np.sum(correct_logprobs) / m
    return loss, probs

def softmax_cross_entropy_backward(probs, y_true):
    m = y_true.shape[0]
    grad = probs
    grad[range(m), y_true] -= 1
    grad /= m
    return grad

# XOR data
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0, 1, 1, 0])  # Labels

# Model: 2 -> 4 -> 2
model = MLP([
    Dense(2, 4),
    ReLU(),
    Dense(4, 2),
    Softmax()
])

# Train
epochs = 5000
lr = 0.05

for epoch in range(epochs):
    logits = model.forward(X)
    loss, probs = cross_entropy_loss(logits, y)
    grad = softmax_cross_entropy_backward(probs, y)
    model.backward(grad, lr)

    if epoch % 100 == 0:
        preds = model.predict(X)
        acc = np.mean(preds == y)
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Acc: {acc:.2f}")
