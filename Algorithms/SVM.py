import numpy as np
def initialize_data():
    X = [[2, 3], [3, 3], [1, 1], [2, 1]]
    y = [1, 1, -1, -1]
    return (X, y)

def compute_gram_matrix(X): 
    N = len(X)
    res = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            res[i][j] = np.dot(X[i], X[j])
            
    return res

def objective_function(gram_mtx, y, lambdas):
    N = len(gram_mtx)
    first_sum = sum(lambdas)
    double_sum = 0
    
    for i in range(N):
        for j in range(N):
            double_sum += lambdas[i] * lambdas[j] * y[i] * y[j] * gram_mtx[i][j]
            
    return first_sum - (0.5 * double_sum)
    
def gradient(gram_mtx, y, lambdas):
    N = len(gram_mtx)
    gradients = [0] * N
    
    for i in range(N):
        sum = 0
        for j in range(N):
            sum += lambdas[j] * y[i] * y[j] * gram_mtx[i][j]
        gradients[i] = 1 - sum
    return gradients

def gradient_ascent(gram_mtx, y, lambdas, eta, iterations, tol=1e-4):
    print("Starting Gradient Ascent...")
    prev_obj_value = float('-inf')
    
    for i in range(iterations):
        gradients = gradient(gram_mtx, y, lambdas)
        for j in range(len(gram_mtx)):
            lambdas[j] += eta * gradients[j]
            lambdas[j] = max(0, lambdas[j])
            
        sum_lambdas_y = sum(lambdas[i] * y[i] for i in range(len(lambdas)))
        correction = sum_lambdas_y / len(lambdas)
        for j in range(len(lambdas)):
            lambdas[j] -= correction * y[j]

            
        obj_value = objective_function(gram_mtx, y, lambdas)
        print(f"Iteration {i + 1}: Objective = {obj_value:.4f}")
        
        # Check for convergence (if the objective value has not improved much)
        if abs(obj_value - prev_obj_value) < tol:
            print(f"Converged after {i + 1} iterations.")
            break
        prev_obj_value = obj_value
    return lambdas

def calculate_bias(gram_mtx, X, y, lambdas, threshold=1e-6):
    support_indices = [i for i, l in enumerate(lambdas) if l > threshold]
    if not support_indices:
        raise ValueError("No vectors found. check lambdas")
    
    b_values = []
    
    for i in support_indices:
        y_i = y[i]
        sum_term = sum(
            lambdas[j] * y[j] * np.dot(X[j], X[i]) for j in range(len(lambdas))
        )
        b = y_i - sum_term
        b_values.append(b)
    return np.mean(b_values)
        
def main():
    # Code to be executed when the script is run directly
    X, y = initialize_data()
    lambdas = [0] * len(X)
    gram_mtx = compute_gram_matrix(X)
    for row in gram_mtx:
        print(row)
        
    lambdas = [0.5, 0.5, 0.2, 0.2]
    gradients = gradient(gram_mtx, y, lambdas)
    print(gradients)
    
    final_lambdas = gradient_ascent(gram_mtx, y, lambdas, 0.05, 100)
    print(final_lambdas)
    b = calculate_bias(gram_mtx, X, y, lambdas)
    print(f"Calculated bias (b): {b}")

if __name__ == "__main__":
    main()