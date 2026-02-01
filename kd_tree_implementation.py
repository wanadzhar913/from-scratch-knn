class KDNode:
    """
    A node in the KDTree.
    """
    def __init__(self, point, k, left=None, right=None):
        self.point = point  # k-dimensional point
        self.k = k  # axis of comparison
        self.left = left  # left subtree
        self.right = right  # right subtree


def build_kd_tree(points: list[tuple[float, float]], depth: int = 0) -> KDNode:
    """
    Build a KDTree from a list of points.

    Args:
        points: list[tuple[float, float]]
        depth: int = 0
    Returns:
        KDNode: The root of the KDTree.
    """
    if not points:
        return None

    k = len(points[0])  # Number of dimensions
    axis = depth % k  # The axis to compare on (% is the modulus operator e.g., 5 % 2 = 1)
    sorted_points = sorted(points, key=lambda p: p[axis])
    median_idx = len(sorted_points) // 2  # // is the floor division operator e.g., 5 // 2 = 2
    median_point = sorted_points[median_idx]  # The median point is the point in the middle of the sorted list

    node = KDNode(median_point, axis)  # Create a new node with the median point and the axis of comparison
    node.left = build_kd_tree(sorted_points[:median_idx], depth + 1)  # Build the left subtree
    node.right = build_kd_tree(sorted_points[median_idx + 1:], depth + 1)  # Build the right subtree

    return node  # Return the root of the KDTree


def distance_squared(point1: tuple[float, float], point2: tuple[float, float]) -> float:
    """
    Calculate the squared distance between two points.

    Args:
        point1: tuple[float, float]
        point2: tuple[float, float]
    Returns:
        float: The squared distance between the two points.
    """
    return sum((x - y) ** 2 for x, y in zip(point1, point2))


def nearest_neighbor(root: KDNode, target: tuple[float, float], depth: int = 0, best: KDNode = None) -> KDNode:
    """
    Find the nearest neighbor of a target point in a KDTree.

    Args:
        root: KDNode
        target: tuple[float, float]
        depth: int = 0
        best: KDNode = None
    Returns:
        KDNode: The nearest neighbor.
    """
    if root is None:
        return best

    k = len(target)
    axis = depth % k

    next_best = None
    next_branch = None

    if best is None or distance_squared(target, root.point) < distance_squared(target, best.point):
        next_best = root
    else:
        next_best = best

    if target[axis] < root.point[axis]:
        next_branch = root.left
        other_branch = root.right
    else:
        next_branch = root.right
        other_branch = root.left

    best = nearest_neighbor(next_branch, target, depth + 1, next_best)
    if distance_squared(target, best.point) > abs(target[axis] - root.point[axis]) ** 2:
        best = nearest_neighbor(other_branch, target, depth + 1, best)

    return best


def main():
    # Example usage
    points = [(3, 6), (17, 15), (13, 15), (6, 12), (9, 1), (2, 7)]
    root = build_kd_tree(points)

    target_point = (9, 2)
    nearest = nearest_neighbor(root, target_point)
    print("Nearest neighbor:", nearest.point)


if __name__ == "__main__":
    main()
