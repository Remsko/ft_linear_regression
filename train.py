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


def gradient_descent(theta0, theta1, x, y, alpha=0.01, cycle=10000):
    m = len(x)
    cost_history = []
    for _ in range(cycle):
        prediction = predict(theta0, theta1, x)
        tmp_theta0 = alpha * (1 / m) * sum(prediction - y)
        tmp_theta1 = alpha * (1 / m) * sum((prediction - y) * x)
        theta0 -= tmp_theta0
        theta1 -= tmp_theta1
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
    print(theta0)
    print(theta1)

    figure = plt.figure('Linear Regression', figsize=(10, 10))
    plt.gcf().subplots_adjust(left=0.09, bottom=0.07,
                              right=0.96, top=0.96, wspace=0, hspace=0.25)

    axes = figure.add_subplot(2, 1, 1)
    axes.set_xlabel('iteration')
    axes.set_ylabel('cost')
    axes.set_title('Cost history')
    cycles = list(range(len(cost_history)))
    axes.plot(cycles, cost_history, color='green')

    axes = figure.add_subplot(2, 1, 2)
    axes.set_xlabel(columns[0])
    axes.set_ylabel(columns[1])
    axes.set_title('Prediction of car price given their mileages')
    axes.scatter(norm_x, norm_y, label='Scatter Plot', color='red')
    axes.plot(norm_x, predict(theta0, theta1, norm_x), label='Regression Line')
    axes.legend()

    plt.show()
