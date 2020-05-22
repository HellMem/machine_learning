from sklearn.cluster import KMeans
import numpy as np


# Based on https://www.youtube.com/watch?v=hDmNF9JG3lo&t=261s

def sklearn_kmeans(X, cluster_size):
    result = KMeans(n_clusters=cluster_size, random_state=0).fit(X)
    return result.labels_, result.cluster_centers_


def calculate_closer_centroid(x_i, clusters_centroids):
    dif_list = []
    for centroid in clusters_centroids:
        dif_list.append(np.sum(np.power(x_i - centroid, 2)))

    dif_list = np.array(dif_list)
    return np.where(dif_list == np.amin(dif_list))[0].flat[0]


def calculate_centroid_position(k, c, data):
    c_indices = np.where(c == k)[0]
    data_assigned = data[c_indices]
    return np.mean(data_assigned, axis=0)


def kmeans(data, clusters, iterations=20):
    data_shape = data.shape[1]
    c_size = data.shape[0]
    c = np.zeros(c_size)
    clusters_centroids = []
    for i in range(clusters):
        clusters_centroids.append(np.random.randint(10, size=data_shape))

    for it in range(iterations):
        for i in range(c_size):
            c[i] = calculate_closer_centroid(data[i], clusters_centroids)

        for k in range(clusters):
            clusters_centroids[k] = calculate_centroid_position(k, c, data)

    return c, clusters_centroids


def run_helper_functions_tests(data):
    clusters_centroids = []
    for i in range(4):
        clusters_centroids.append(np.random.randint(100, size=data.shape[1]))
    print(clusters_centroids)
    print(calculate_closer_centroid(np.array([75, 23]), clusters_centroids))

    print(calculate_centroid_position(1, np.array([1, 1, 2]), np.array([[1, 2], [2, 2], [1, 1]])))


if __name__ == "__main__":
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    np.random.seed(1234)
    labels, centroids = kmeans(X, 2)
    print('Implementation')
    print(labels)
    print(centroids)
    print('-' * 100)
    labels, centroids = sklearn_kmeans(X, 2)
    print('SKlearn')
    print(labels)
    print(centroids)

