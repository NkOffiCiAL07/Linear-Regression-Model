import numpy as np
import csv

def import_data():
    X = np.genfromtxt("train_X_lr.csv", delimiter=',', dtype=np.float64, skip_header=1)
    Y = np.genfromtxt("train_Y_lr.csv", delimiter=',', dtype=np.float64)
    return X, Y

def compute_gradient_of_cost_function(X, Y, W):
    Y_pred = np.dot(X, W)
    difference = Y_pred - Y
    dw = (1/len(X))*(np.dot(difference.T, X))
    dw = dw.T
    return dw

def optimize_weight_using_gradient_descent(X, Y, W, number_of_interations, learning_rate):
    for i in range(number_of_interations):
        dw = compute_gradient_of_cost_function(X, Y, W)
        W = W - (learning_rate*dw)
    return W

def train_model(X, Y):
    X = np.insert(X, 0, 1, axis=1)
    Y = Y.reshape(len(X), 1)
    W = np.zeros((X.shape[1], 1))
    W = optimize_weight_using_gradient_descent(X, Y, W, 10**8, 0.0002)
    return W

def save_model(weights, weight_file_name):
    with open(weight_file_name, "w") as weights_file:
        wr = csv.writer(weights_file)
        wr.writerows(weights)
        weights_file.close()

if __name__=="__main__":
    X, Y = import_data()
    weights = train_model(X, Y)
    save_model(weights, "WEIGHTS_FILE.csv")

