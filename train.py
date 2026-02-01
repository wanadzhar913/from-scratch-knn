import time

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from models import KNNClassification, KNNClassificationFast

cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

start_time = time.time()
knn = KNNClassification(k=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
end_time = time.time()
time_taken_knn = end_time - start_time
accuracy_knn = np.sum(y_test == predictions) / len(y_test)
print(f"Accuracy for KNNClassification: {accuracy_knn:.2f}\n")
print(f"Time taken for KNNClassification: {time_taken_knn:.2f} seconds\n")

start_time = time.time()
knn = KNNClassificationFast(k=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
end_time = time.time()
time_taken_kd_tree = end_time - start_time
accuracy_kd_tree = np.sum(y_test == predictions) / len(y_test)
print(f"Accuracy for KNNClassification with KDTree implementation: {accuracy_kd_tree:.2f}\n")
print(f"Time taken for KNNClassification with KDTree implementation: {time_taken_kd_tree:.2f} seconds\n")

print("Plotting the KNN classification results")
plt.figure(figsize=(12, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=predictions, cmap=cmap, edgecolors='k', s=20, label='Predicted Labels')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('KNN Classification')
plt.savefig('knn_classification.png')
plt.close()

