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


# K-Means
def k_means(data, k=2):
    if not isinstance(k, int) or k <= 0 or len(data) < k:
        return

    # Select first K points as centroids
    centroids = {0: data[0], 1: data[1]}

    # configurations
    limit = 0.0001
    max_loop_count = 300
    total_steps = []
    # Loop
    for i in range(max_loop_count):
        # Classification data into K groups
        groups = {}

        for j in range(k):
            groups[j] = []

        for item in data:
            dist = [np.linalg.norm(centroids[centroid] - item) for centroid in centroids]
            index = dist.index(min(dist))
            groups[index].append(item)

        # Calculate new centroids
        new_centroids = [np.average(groups[i], axis=0) for i in groups]
        # Store data for matplotlib
        total_steps.append({
            'loop': i,
            'groups': groups,
            'centroids': centroids.copy()
        })

        # Check whether they change or not
        stop_loop = True
        for c in centroids:
            if abs(np.sum((new_centroids[c] - centroids[c])/centroids[c]*100.0)) > limit:
                stop_loop = False
                break

        if stop_loop:
            break

        # Update centroids
        for c in centroids:
            centroids[c] = new_centroids[c]

    # Draw pictures
    colors = k*['g', 'r', 'b', 'c', 'm', 'y', 'k', 'w']
    fig = plt.figure()
    for step in total_steps:
        # This may cause error if len(total_steps) > 9
        ax = fig.add_subplot(1, len(total_steps), step['loop'] + 1)
        for g in step['groups']:
            for point in step['groups'][g]:
                ax.scatter(point[0], point[1], s=20, color=colors[g])
            ax.scatter(step['centroids'][g][0], step['centroids'][g][1], marker='x', s=30, color=colors[g])
    plt.show()


k_means(X)
