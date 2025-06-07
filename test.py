from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression
import pandas as pd

df = pd.read_csv("test.csv")
X=df[["x"]]
y = df["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
n_iter=int(input("n_iter: "))
fig = plt.figure(figsize=(8,6))

reg = LinearRegression(n_iter=n_iter)

reg.fit(X_train, y_train)

predictions = reg.predict(X_test)

print(predictions)


def mse(y_test, predictions):
    return np.mean((y_test-predictions)**2)

mse = mse(y_test, predictions)

print(mse)

test_x = np.array([[-1.7]])


y_pred = reg.predict(test_x)
print(y_pred)

plt.plot([n for n in range(n_iter)], reg.loss_hist)
plt.xlabel("number of iterations")
plt.ylabel("L2")
plt.title("Loss function convergence(L2) curve")
plt.show()
plt.scatter(X,y)
plt.plot(X, reg.predict(X),color="red")
plt.xlabel("X points from test dataset")
plt.ylabel("Y points from test dataset and model predictions")
plt.title()

plt.show()
"""For the current dataset used, the model converges at around the 300th iteration, feel free to change the datasets or use 
datasets.make_regression() from sklearn to tweak it and test (the model is compatible with the sklearn generated regression dataset)"""