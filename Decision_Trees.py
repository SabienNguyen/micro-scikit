import numpy as np

def entropy(y): 
    classes, counts = np.unique(y, return_counts=True)
    
    prob = counts / len(y)
    
    return -np.sum(prob * np.log2(prob))


def information_gain(X, y, feature_index):
    #unique values of features
    feature_values = np.unique(X[:, feature_index])
    
    #entropy of dataset
    parent_entropy = entropy(y)
    
    weighted_entropy_sum = 0
    
    for value in feature_values:
        subset_mask = X[:, feature_index] == value
        y_subset = y[subset_mask]
        
        subset_entropy = entropy(y_subset) 
        
        weighted_entropy_sum += (len(y_subset) / len(y)) * subset_entropy
        
    return parent_entropy - weighted_entropy_sum

def best_split(X, y):
    best_gain = -float('inf')  # Initialize the best gain as a very low value
    best_feature = None        # To store the best feature index
    
    for feature_index in range(len(X[0])):  # Loop through each feature
        info_gain =  information_gain(X, y, feature_index)
        
        # 2. If this gain is better than the best one seen so far, update best_gain and best_feature
        if info_gain > best_gain:
            best_gain =  info_gain
            best_feature = feature_index
    return best_feature  # Return the best feature index
y = np.array([1, 1, 0, 0, 1, 1, 0])
print("Entropy of y:", entropy(y))

# Test with a simple dataset
X = np.array([[1, 2], [1, 3], [2, 2], [2, 3], [3, 2], [3, 3]])  # Sample features
y = np.array([0, 0, 1, 1, 0, 1])  # Labels corresponding to the features
print("Information gain for feature 0:", information_gain(X, y, 0))  # Information gain for first feature
print("Information gain for feature 1:", information_gain(X, y, 1))  # Information gain for second feature