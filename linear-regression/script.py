import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from linear_regression import LinearRegression
from sklearn.metrics import mean_squared_error as mse

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression(lr=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(mse(y_test, y_pred))


cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
y_pred = model.predict(X)
plt.plot(X, y_pred, color='black', linewidth=2, label="Prediction")
plt.show()
