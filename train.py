import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def average(x):
    return sum(x) / len(x)

def normalize(x):
    return (x - average(x)) / (max(x) - min(x))

def predict(theta0, theta1, x):
    prediction = theta0 + theta1 * x
    return prediction
    

def cost(theta0, theta1, x, y):
    m = len(x)
    cost = sum((predict(theta0, theta1, x) - y) ** 2) / (2 * m)
    return cost

def gradient_descent(theta0, theta1, x, y, alpha=0.001, cycle=1000):
    m = len(x)
    cost_history = []
    for _ in range(cycle):
        tmp_theta0 = alpha * (1 / m) * sum(predict(theta0, theta1, x) - y)
        tmp_theta1 = alpha * (1 / m) * sum(predict(theta0, theta1, x) - y) * x
        theta0 = tmp_theta0
        theta1 = tmp_theta1
        cost_history.append(cost(theta0, theta1, x, y))
    return theta0, theta1, cost_history

if __name__ == "__main__":
    data = pd.read_csv('data.csv')
    columns = data.columns.values[:2]

    x = np.array(data[columns[0]].values)
    y = np.array(data[columns[1]].values)

    norm_x = normalize(x)
    norm_y = normalize(y)


    theta0, theta1, cost_history = gradient_descent(0, 0, norm_x, norm_y)

    #plt.plot(list(range(len(cost_history))), cost_history, label='Regression Line')
    plt.plot(x, predict(theta0, theta1, x), label='Regression Line')

    #plt.scatter(x, y, label='Scatter Plot')    
    plt.scatter(norm_x, norm_y, label='Scatter Plot')
    
    # plt.xlabel(columns[0])
    # plt.ylabel(columns[1])
    # plt.legend()
    plt.show()