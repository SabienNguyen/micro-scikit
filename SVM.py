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
    

def main():
    # Code to be executed when the script is run directly
    X, y = initialize_data()
    lambdas = [0 * len(X)]
    gram_mtx = compute_gram_matrix(X)
    for row in gram_mtx:
        print(row)

if __name__ == "__main__":
    main()