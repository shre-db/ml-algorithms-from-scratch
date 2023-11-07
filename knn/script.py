import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
iris_X, iris_y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.2, random_state=0)

from knn import KNN

knn_clf = KNN(k=1)
knn_clf.fit(X_train, y_train)
y_pred = knn_clf.predict(X_test)

score = accuracy_score(y_test, y_pred)
print(score)
