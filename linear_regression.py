import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)

#Import CSV with pandas
data = pd.read_csv("data.csv")
print(data.shape)
print(data)

#Get X and Y
X = data["km"].values
Y = data["price"].values

#Get means
mean_x = np.mean(X)
mean_y = np.mean(Y)

#Get number of datas
m = len(X)

#Calculate Coeficient
numer = 0
denom = 0
for i in range(m):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2
b1 = numer / denom
b0 = mean_y - (b1 * mean_x)

print(b1, b0)

#Plot

max_x = np.max(X) + 100
min_x = np.min(X) - 100

potit_x = np.linspace(min_x, max_x, 1000)
potit_y = b0 + b1 * potit_x

plt.plot(potit_x, potit_y, color= "#58b970", label="Regression Line")
plt.scatter(X, Y, c="#ef5423", label="Scatter Plot")

plt.xlabel("km")
plt.ylabel("price")

plt.legend()
plt.show()

if __name__ == "__main__":
    pass