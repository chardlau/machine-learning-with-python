import numpy as np
import matplotlib.pyplot as plt

# Input data set
X = np.array([
    [-4, -3.5], [-3.5, -5], [-2.7, -4.5],
    [-2, -4.5], [-2.9, -2.9], [-0.4, -4.5],
    [-1.4, -2.5], [-1.6, -2], [-1.5, -1.3],
    [-0.5, -2.1], [-0.6, -1], [0, -1.6],
    [-2.8, -1], [-2.4, -0.6], [-3.5, 0],
    [-0.2, 4], [0.9, 1.8], [1, 2.2],
    [1.1, 2.8], [1.1, 3.4], [1, 4.5],
    [1.8, 0.3], [2.2, 1.3], [2.9, 0],
    [2.7, 1.2], [3, 3], [3.4, 2.8],
    [3, 5], [5.4, 1.2], [6.3, 2]
])


def mean_shift(data, radius=2):
    # Cheat each data point as a centroid
    centroids = {}
    for i in range(len(data)):
        centroids[i] = data[i]

    while True:
        new_centroids = []
        for i in centroids:
            in_circle = []
            centroid = centroids[i]
            for j in centroids:
                if np.linalg.norm(centroids[j] - centroid) < radius:
                    in_circle.append(centroids[j])
            # Calculate new centroid
            new_centroid = np.average(in_circle, axis=0)
            new_centroids.append(tuple(new_centroid))

        # Remove redundant centroids
        uniques = sorted(list(set(new_centroids)))

        # Update centroids
        old_centroids = dict(centroids)
        new_centroids = {}
        for i in range(len(uniques)):
            new_centroids[i] = np.array(uniques[i])
        centroids = new_centroids

        show_centroids(old_centroids)

        print('old_centroids: ', old_centroids)
        print('new_centroids: ', new_centroids)
        # Check should we stop or not
        stop_loop = True
        for i in new_centroids:
            # The length of prev_centroids must large or equal then the one of centroids
            # So there is not any problem
            if not np.array_equal(new_centroids[i], old_centroids[i]):
                stop_loop = False
                break

        if stop_loop:
            break

    return centroids


def show_centroids(centroids):
    plt.figure(figsize=(5, 5))
    plt.xlim((-8, 8))
    plt.ylim((-8, 8))
    plt.scatter(X[:, 0], X[:, 1], s=20)
    theta = np.linspace(0, 2 * np.pi, 800)
    for c in centroids:
        plt.scatter(centroids[c][0], centroids[c][1], color='r', marker='x', s=30)
        x, y = np.cos(theta) * radius + centroids[c][0], np.sin(theta) * radius + centroids[c][1]
        plt.plot(x, y, linewidth=1)
    plt.show()


radius = 2
mean_shift(X, radius=radius)
