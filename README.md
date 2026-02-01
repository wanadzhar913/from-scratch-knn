### Objectives
We implement the [K-Nearest Neighbour](https://www.geeksforgeeks.org/machine-learning/k-nearest-neighbours/) algorithm from scratch. We detail the algorithm below.

Given a data point:
- Calculate its distance from all other data points in the dataset
- Get the closest K points
- The next steps differ depending on the problem:
    - **Regression:** Get the average of distance betweet the data point and it's *k* neighbours
    - **Classification:** Get the label with majority vote (if it's closest *k* neighbours are of class green, then it's also green)

### To start the project
We use [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for package management.

```bash
git clone https://github.com/wanadzhar913/from-scratch-knn.git
cd from-scratch-knn

uv venv
uv sync

python3 train.py
```

### Formulas & Data Structures

[Euclidean distance](https://www.datacamp.com/tutorial/euclidean-distance) is given by the below:

```math
d(x, y) = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2 + ... + (x_n - y_n)^2}
```

After looking at [scikit-learn's implementation](https://github.com/scikit-learn/scikit-learn/blob/d3898d9d5/sklearn/neighbors/_classification.py#L71-L78), we can see they make use of several interesting data structures.
Here, we test for differences using the [KDTree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html), instead of brute force.

We also use Python's `Counter` data structure:

```python
from collections import Counter

labels = ['green', 'red', 'green', 'red', 'green']

Counter(labels).most_common(1)

>>> [('green', 3)]
```

### Resources
- KDTree implementation in Python: https://medium.com/@isurangawarnasooriya/exploring-kd-trees-a-comprehensive-guide-to-implementation-and-applications-in-python-3385fd56a246
- KNN-implementation step-by-step: https://youtu.be/rTEtEy5o3X0?si=KamJLclr2qKkLaaA