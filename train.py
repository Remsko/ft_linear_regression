import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

g_cost_history = []
g_theta_history = []


def average(x):
    return sum(x) / len(x)


def extremum_sub(x):
    return max(x) - min(x)


def mean_normalize(x):
    return (x - average(x)) / extremum_sub(x)


def mean_reverse(theta0, theta1, x):
    theta0 = theta0 - theta1 * average(x) / extremum_sub(x)
    theta1 = theta1 / extremum_sub(x)
    return theta0, theta1


def predict(theta0, theta1, x):
    prediction = theta0 + theta1 * x
    return prediction


def cost(theta0, theta1, x, y):
    m = len(x)
    cost = sum((predict(theta0, theta1, x) - y) ** 2) / (2 * m)
    return cost


def gradient_descent(theta0, theta1, x, y, alpha=0.1, cycle=1000, step=10):
    m = len(x)
    for i in range(cycle):
        prediction = predict(theta0, theta1, x)
        tmp_theta0 = alpha * (1 / m) * sum(prediction - y)
        tmp_theta1 = alpha * (1 / m) * sum((prediction - y) * x)
        theta0 -= tmp_theta0
        theta1 -= tmp_theta1
        g_cost_history.append(cost(theta0, theta1, x, y))
        if i % (cycle / step) == 0:
            g_theta_history.append([theta0, theta1])
    return theta0, theta1


def cost_history_display(figure):
    cycles = list(range(len(g_cost_history)))
    # Split
    if "-v" in sys.argv or "-l" in sys.argv or "-h" in sys.argv:
        axes = figure.add_subplot(2, 1, 1)
    else:
        axes = figure.add_subplot(1, 1, 1)
    # Meta
    axes.set_xlabel('iteration')
    axes.set_ylabel('cost')
    axes.set_title('Cost Evolution')
    # Cost Evolution
    axes.plot(cycles, g_cost_history, color='green')


def linear_regression_display(figure, columns, x, y, theta0, theta1):
    # Split
    if "-c" in sys.argv:
        axes = figure.add_subplot(2, 1, 2)
    else:
        axes = figure.add_subplot(1, 1, 1)
    # Meta
    axes.set_xlabel(columns[0])
    axes.set_ylabel(columns[1])
    axes.set_title('Prediction of car price given their mileages')
    # Data display
    axes.scatter(x, y, label='Scatter Plot', color='blue')
    # Regression Line
    if "-l" in sys.argv:
        axes.plot(x, predict(theta0, theta1, x),
                  label='Regression Line', color='#840606')
    # Regression evolution
    if "-h" in sys.argv:
        for theta in g_theta_history:
            theta0, theta1 = mean_reverse(theta[0], theta[1], x)
            axes.plot(x, predict(theta0, theta1, x), color='#84060688')
    axes.legend()


def train(data):
    # Entries
    columns = data.columns.values[:2]
    x = np.array(data[columns[0]].values)
    y = np.array(data[columns[1]].values)
    # Treatement
    norm_x = mean_normalize(x)
    theta0, theta1 = gradient_descent(0, 0, norm_x, y)
    theta0, theta1 = mean_reverse(theta0, theta1, x)
    return x, y, theta0, theta1


def create_displayer():
    figure = plt.figure('Linear Regression', figsize=(10, 10))
    plt.gcf().subplots_adjust(left=0.09, bottom=0.07,
                              right=0.96, top=0.96, wspace=0, hspace=0.25)
    return figure


def display(data, x, y, theta0, theta1):
    figure = create_displayer()
    # Cost Evolution
    if "-c" in sys.argv:
        cost_history_display(figure)
    # Graph
    if "-v" in sys.argv or "-l" in sys.argv or "-h" in sys.argv:
        columns = data.columns.values[:2]
        linear_regression_display(figure, columns, x, y, theta0, theta1)
    plt.show()

def save(theta0, theta1):
    snapshot = {
        "theta0": theta0,
        "theta1": theta1
    }
    with open('snapshot.txt', 'w') as outfile:
        json.dump(snapshot, outfile)

def main():
    # Parsing
    try:
        data = pd.read_csv('data.csv')
    except:
        print("Failed to parse dataset !")
        return -1
    # Training
    x, y, theta0, theta1 = train(data)
    save(theta0, theta1)
    # Display
    if "-c" in sys.argv or "-v" in sys.argv or "-l" in sys.argv or "-h" in sys.argv:
        display(data, x, y, theta0, theta1)


if __name__ == "__main__":
    main()
