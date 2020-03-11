import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_vectors_from_csv(filename):
    """
    Retrieve the km and the price row and assign and return it
    """
    
    #Import CSV with pandas
    data = pd.read_csv(filename)

    #Get X and Y
    Xrow = data["km"].values
    Yrow = data["price"].values
    return Xrow, Yrow


def rescale_vector(vector):
    """
    Reshape the vector to get his mean at 0 and his range between -1 and 1
    """
    
    #Get average and range
    vector_avg = np.sum(vector) / len(vector)
    vector_range = np.max(vector) - np.min(vector)
    
    #Center at 0
    vector = vector - vector_avg
    #get range to [-1 : 1]
    vector = vector / vector_range
    
    return vector


def gradient_descent(X, Y):
    """
    Perform a gradient descent between the input X and the output Y
    """
    
    m = 0
    b = 0
    n = len(X)
    iterations = 100000
    alpha=0.001

    for i in range(iterations):
        y_predicted = m * X + b
        cost = (1/n) * sum([val**2 for val in (Y-y_predicted)])
        md = -(2/n)*sum(X*(Y-y_predicted))
        bd = -(2/n)*sum(Y-y_predicted)
        m = m - alpha * md
        b = b - alpha * bd
    print("m {}, b {}, cost {}, iteration {}".format(m, b, cost, i))
    
    return [b, m]


def unscale_theta(theta_0, theta_1, x):
    """
    "Unscales" the theta found.
    If we don't unscale it, predictions will be kinda wrong cause scaled down.
    """

    x_avg = sum(x) / len(x)
    x_range = max(x) - min(x)
    theta_0 = theta_0 - theta_1 * x_avg / x_range
    theta_1 = theta_1 / (max(x) - min(x))
    return [theta_0, theta_1]



def plot_dataset(x, y, theta):
    """
    Plots the dataset from x and y.
    """

    plt.figure("Prices of cars given their mileages")
    plt.plot(x, [(theta[0] + theta[1] * i) for i in x], color="r")
    plt.scatter(x, y, color="g")
    plt.xlabel("Mileages")
    plt.ylabel("Prices")
    plt.show()


def main():
    """
    Main function
    """

    #Extract vectors from csv
    X, Y = get_vectors_from_csv("data.csv")

    theta = [0.0, 0.0]
    #Rescale X because his range is too wide
    theta = gradient_descent(rescale_vector(X), Y)
    #Unscale the theta to make it valid for the real values
    theta = unscale_theta(theta[0], theta[1], X)

    plot_dataset(X, Y, theta)

if __name__ == "__main__":
    main()