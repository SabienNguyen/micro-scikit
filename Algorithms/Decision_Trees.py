import numpy as np
from collections import Counter


class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
def split_data(X, y, feature_index, threshold):
    left_X, left_y = [], []
    right_X, right_y = [], []
    
    for i, row in enumerate(X):
        if row[feature_index] <= threshold:
            left_X.append(row)
            left_y.append(y[i])
        else:
            right_X.append(row)
            right_y.append(y[i])
    
    return left_X, left_y, right_X, right_y

def most_common_label(y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common

def build_tree(X, y, depth=0, max_depth=5):
    if len(set(y)) == 1:
        return TreeNode(value=y[0])
    
    if max_depth is not None and depth >= max_depth:
        return TreeNode(value=most_common_label(y))
    
    best_feature, threshold = best_split(X, y)
    
    if best_feature is None:
        return TreeNode(value=most_common_label(y))
    
    left_X, left_y, right_X, right_y = split_data(X, y, best_feature, threshold)
    
    left_subtree = build_tree(left_X, left_y, depth + 1, max_depth)
    right_subtree = build_tree(right_X, right_y, depth + 1, max_depth)
    
    return TreeNode(feature=best_feature, threshold=threshold, left=left_subtree, right=right_subtree)

def print_tree(node, spacing=""):
    # Base case: If this is a leaf node, print the label
    if node.value is not None:
        print(spacing + f"Leaf: {node.value}")
        return
    
    # Print the feature and threshold
    print(spacing + f"Feature {node.feature} <= {node.threshold}")
    
    # Recur for left and right branches
    print(spacing + "--> Left:")
    print_tree(node.left, spacing + "  ")
    
    print(spacing + "--> Right:")
    print_tree(node.right, spacing + "  ")

def entropy(y): 
    classes, counts = np.unique(y, return_counts=True)
    
    prob = counts / len(y)
    
    return -np.sum(prob * np.log2(prob))


def information_gain(X, y, feature_index, threshold):    
    #entropy of dataset
    parent_entropy = entropy(y)
    
    left_y = [y[i] for i in range(len(X)) if X[i][feature_index] <= threshold]
    right_y = [y[i] for i in range(len(X)) if X[i][feature_index] > threshold]
    
    left_weight = len(left_y) / len(y)
    right_weight = len(right_y) / len(y)
    
    weighted_entropy = left_weight * entropy(left_y) + right_weight * entropy(right_y)
    
    info_gain = parent_entropy - weighted_entropy
    return info_gain

def predict(tree, sample):
    if isinstance(tree, str):  # Tree is a label
        return tree

    # Extract the feature and threshold for the current decision node
    feature, subtree = list(tree.items())[0]
    feature_index = int(feature.split()[1])  # Extract the feature index from the text

    # Split the sample based on the feature threshold
    if sample[feature_index] <= subtree["threshold"]:
        return predict(subtree["left"], sample)
    else:
        return predict(subtree["right"], sample)

# Sample usage:
tree = {
    "Feature 0": {
        "threshold": 2,
        "left": "Apple",
        "right": "Orange"
    }
}

sample = [1.5, 2]  # Example input
print(predict(tree, sample))  # Expected: "Apple"

def best_split(X, y):
    best_gain = -float('inf')
    best_feature = None
    best_threshold = None
    
    for feature_index in range(len(X[0])):  # Loop through each feature
        unique_values = set([row[feature_index] for row in X])  # Unique values of the feature
        
        for threshold in unique_values:  # Test each unique value as a threshold
            info_gain = information_gain(X, y, feature_index, threshold)
            
            if info_gain > best_gain:
                best_gain = info_gain
                best_feature = feature_index
                best_threshold = threshold
    
    return best_feature, best_threshold

# Sample Dataset
X = [
    [2, 3],
    [1, 5],
    [2, 4],
    [3, 2],
    [5, 2],
    [4, 5]
]

y = ["Apple", "Apple", "Apple", "Orange", "Orange", "Orange"]

# Build the Tree
tree = build_tree(X, y, max_depth=3)

# Visualize the Tree
print("Decision Tree Structure:")
print_tree(tree)

tree = {
    "Feature 0": {
        "threshold": 2,
        "left": "Apple",
        "right": "Orange"
    }
}

sample = [1.5, 2]  # Example input
print(predict(tree, sample))  # Expected: "Apple"