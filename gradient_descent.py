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
        # if "-error" in sys.argv:
        #     plt.text(
        #         150000, 8000,
        #         f"Estimated error: {estimate_error_percent(x, y, theta)}%",
        #         bbox=dict(facecolor="lightblue", alpha=0.5)
        #     )
    plt.show()


#Import CSV with pandas
data = pd.read_csv("data.csv")
print(data.shape)
print(data)

#Get X and Y
X_raw = data["km"].values
Y_raw = data["price"].values

X = preprocessing.scale(X_raw)
Y = preprocessing.scale(Y_raw)

#X = np.array([1,2,3,4,5])
#Y = np.array([5,7,9,11,13])

theta = [0.0, 0.0]

theta[0], theta[1] = gradient_descent(X, Y)

theta[0] = theta[0] * np.var(Y_raw) + np.mean(Y_raw)
theta[1] = theta[1] * np.var(Y_raw) / np.var(X_raw)

print(theta[0])
print(theta[1])
plot_dataset(X_raw, Y_raw, theta)

#Plot


# max_x = np.max(X)
# min_x = np.min(X)

# print ("minX {}, maxX {}, minY {}, maxY {}".format(min_x, max_x, np.min(Y), np.max(Y)))

# potit_x = np.linspace(min_x, max_x, 1000)
# potit_y = b + m * potit_x

# plt.plot(potit_x, potit_y, color= "#58b970", label="Regression Line")
# plt.scatter(X, Y, c="#ef5423", label="Scatter Plot")

# plt.xlabel("km")
# plt.ylabel("price")

# plt.legend()
# plt.show()