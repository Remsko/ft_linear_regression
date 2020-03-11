import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


#plt.rcParams['figure.figsize'] = (10.0, 10.0)

def gradient_descent(X, Y):
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
    
    return b, m


def plot_dataset(x, y, theta):
    """
    Plots the dataset from x and y.
    If plot_model is set to True, also plot the trained model.
    """


    plt.figure("Prices of cars given their mileages")

    for i in x:
        print(i)

    plt.plot(x, [(theta[0] + theta[1] * i) for i in x], color="r")
    plt.scatter(x, y, color="g")
    plt.xlabel("Mileages")
    plt.ylabel("Prices")
    plt.show()




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

#Import CSV with pandas
data = pd.read_csv("data.csv")
print(data.shape)
print(data)

#Get X and Y
X = data["km"].values
Y = data["price"].values

#Rescale Data
x_avg = np.sum(X) / len(X)
x_range = np.max(X) - np.min(X)
#Center at 0
X_rs = X - x_avg
#get range to [-1 : 1]
X_rs = X_rs / x_range


#X = np.array([1,2,3,4,5])
#Y = np.array([5,7,9,11,13])

theta = [0.0, 0.0]
theta[0], theta[1] = gradient_descent(X_rs, Y)

print(theta)
theta = unscale_theta(theta[0], theta[1], X)
print(theta)
plot_dataset(X, Y, theta)