from collections import Counter

import numpy as np
from scipy.spatial import KDTree

def majority_vote(labels: list) -> str:
    """
    This is a helper function to determine the majority vote from a list of labels.
    The majority vote is determined by the label that appears most frequently in the list.

    ```python
    from collections import Counter
    labels = ['green', 'red', 'green', 'red', 'green']
    Counter(labels).most_common(1)
    
    >>> [('green', 3)]
    ```

    Args:
        labels: list
    Returns:
        The majority label.
    """
    return Counter(labels).most_common(1)[0][0]


def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    This is a helper function to calculate the Euclidean distance between two points.
    The Euclidean distance is defined as the square root of the sum of the squared differences of the coordinates as per below:

    ```math
    d(x, y) = \\sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2 + ... + (x_n - y_n)^2}
    ```

    Args:
        x1: np.ndarray
        x2: np.ndarray
    Returns:
        The Euclidean distance between the two points.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNNClassification:
    def __init__(self, k: int):
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x: np.ndarray):
        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Return the most common class label
        return majority_vote(k_nearest_labels)


class KNNRegression:
    def __init__(self, k: int):
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x: np.ndarray):
        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Return the average of the k nearest neighbor training samples
        return np.mean(k_nearest_labels)


class KNNClassificationFast:
    """
    This is a faster implementation of the KNN classification algorithm using KDTree.
    """
    def __init__(self, k: int):
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.tree = KDTree(X)
        self.y_train = y

    def predict(self, X: np.ndarray):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x: np.ndarray):
        # Query the KDTree for the k nearest neighbors
        distances, indices = self.tree.query(x, k=self.k)

        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in indices]

        # Return the most common class label
        return majority_vote(k_nearest_labels)


class KNNRegressionFast:
    """
    This is a faster implementation of the KNN regression algorithm using KDTree.
    """
    def __init__(self, k: int):
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.tree = KDTree(X)
        self.y_train = y

    def predict(self, X: np.ndarray):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x: np.ndarray):
        # Query the KDTree for the k nearest neighbors
        distances, indices = self.tree.query(x, k=self.k)

        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in indices]

        # Return the average of the k nearest neighbor training samples
        return np.mean(k_nearest_labels)


if __name__ == "__main__":
    import time

    # Test the majority vote function
    print("Testing the majority vote function")
    labels = ['green', 'red', 'green', 'red', 'green']
    print(Counter(labels).most_common(1)) # [('green', 3)]
    print('--------------------------------\n')

    print("Testing the Counter function")
    print(Counter('gallahad')) # Counter({'a': 3, 'g': 1, 'l': 2, 'h': 1, 'd': 1})
    print('--------------------------------\n')

    print("Testing the Euclidean distance function")
    print(euclidean_distance(np.array([1, 2, 3]), np.array([4, 5, 6]))) # 5.196152422706632
    print('--------------------------------\n')

    print("Testing the KNNClassification class")
    # Test the KNN classes
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(10, 10)
    y_test = np.random.randint(0, 2, 10)

    start_time = time.time()
    knn = KNNClassification(k=3)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    print(predictions)
    end_time = time.time()
    time_taken_knn = end_time - start_time
    print(f"Time taken for KNNClassification: {time_taken_knn:.2f} seconds\n")


    print("Testing the KNNClassificationFast class")
    start_time = time.time()
    knn = KNNClassificationFast(k=3)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    print(predictions)
    end_time = time.time()
    time_taken_kd_tree = end_time - start_time
    print(f"Time taken for KDTreeClassification: {time_taken_kd_tree:.2f} seconds\n")

    print(f"As we can see, the KNNClassificationFast class is much faster than the KNNClassification class by a factorof {time_taken_kd_tree / time_taken_knn:.2f} times (**NOTE:** This is a very small dataset, so the difference is not very significant).")