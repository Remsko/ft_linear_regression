import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def average(x):
    return sum(x) / len(x)


def mean_normalize(x):
    return (x - average(x)) / (max(x) - min(x))


def mean_reverse(theta0, theta1, x):
    theta0 = theta0 - theta1 * average(x) / (max(x) - min(x))
    theta1 = theta1 /(max(x) - min(x))
    return theta0, theta1


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


def cost_history_display(figure, cost_history):
    cycles = list(range(len(cost_history)))

    axes = figure.add_subplot(2, 1, 1)
    axes.set_xlabel('iteration')
    axes.set_ylabel('cost')
    axes.set_title('Cost history')
    
    axes.plot(cycles, cost_history, color='green')


def linear_regression_display(figure, columns, x, y, theta0, theta1):
    axes = figure.add_subplot(2, 1, 2)
    axes.set_xlabel(columns[0])
    axes.set_ylabel(columns[1])
    axes.set_title('Prediction of car price given their mileages')

    axes.scatter(x, y, label='Scatter Plot', color='red')
    axes.plot(x, predict(theta0, theta1, x), label='Regression Line')
    axes.legend()


def create_displayer():
    figure = plt.figure('Linear Regression', figsize=(10, 10))
    plt.gcf().subplots_adjust(left=0.09, bottom=0.07,
                              right=0.96, top=0.96, wspace=0, hspace=0.25)

    return figure


if __name__ == "__main__":
    data = pd.read_csv('data.csv')
    columns = data.columns.values[:2]

    x = np.array(data[columns[0]].values)
    y = np.array(data[columns[1]].values)

    norm_x = mean_normalize(x)

    theta0, theta1, cost_history = gradient_descent(0, 0, norm_x, y)
    theta0, theta1 = mean_reverse(theta0, theta1, x)

    figure = create_displayer()
    cost_history_display(figure, cost_history)
    linear_regression_display(figure, columns,x, y, theta0, theta1)

    plt.show()
