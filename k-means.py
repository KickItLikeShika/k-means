import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat



def plot_clusters(X, idx, centroids):
    """Plot the data and the centroids."""
    # Make the three clusters
    cluster1 = X[np.where(idx == 0)[0], :]
    cluster2 = X[np.where(idx == 1)[0], :]
    cluster3 = X[np.where(idx == 2)[0], :]


    # Draw the first cluster
    plt.scatter(cluster1[:, 0], cluster1[:, 1], s=30, color='r')
    plt.scatter(centroids[0, 0], centroids[0, 1], s=300, color='r')

    # Draw the second cluster
    plt.scatter(cluster2[:, 0], cluster2[:, 1], s=30, color='g')
    plt.scatter(centroids[1, 0], centroids[1, 1], s=300, color='g')

    # Draw the third cluster
    plt.scatter(cluster3[:, 0], cluster3[:, 1], s=30, color='b')
    plt.scatter(centroids[2, 0], centroids[2, 1], s=300, color='b')

    plt.show()



def run_k_means(X, init_centroids, max_iters):
    """Run the algorithm."""
    # Number of data and features
    m, n = X.shape
    # Number of clusters
    k = init_centroids.shape[0]
    # Indecies
    idx = np.zeros(m)
    # Centroids
    centroids = init_centroids

    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = move_centroids(X, idx, k)

    return idx, centroids


def move_centroids(X, idx, k):
    """Move the cenroids to the right new place."""
    # Number of data and features
    m, n = X.shape
    # Initialize the centroids array
    centroids = np.zeros((k, n))

    for i in range(k):
        # Get all the points in the cluster i
        indices = np.where(idx == i)

        # Get the mean and move the centroid to the center of the assigned points
        centroids[i, :] = (np.sum(X[indices, :], axis=1) / len(indices[0])).ravel()

    return centroids



def find_closest_centroids(X, centroids):
    """Get the closest centroid for each point in the data."""
    # Number of elemets
    m = X.shape[0]
    # Number of centroids or clusters
    k = centroids.shape[0]
    # Index of each element in the originat data to 
    # clearify which element belongs to which cluster
    idx = np.zeros(m)

    for i in range(m):
        min_dist = 1000000
        for j in range(k):
            # Subtract each element from each centroid to
            # get the minimum distance between the element
            # and the centroid
            dist = np.sum((X[i, :] - centroids[j, :]) ** 2)

            # If the new distance is less than the minimum distance
            # then the minimum distance is the new one
            if dist < min_dist:
                # Assign the element to its cluster
                min_dist = dist
                idx[i] = j
    
    return idx



def initialize_centroids(X, K):
    """Initialize the centroids with values from our data."""
    # Number of elemets and features
    m, n = X.shape
    # Initialize our centroids array
    centroids = np.zeros((K, n))
    # Random indices in the data
    idx = np.random.randint(0, m, K)

    # Assign each centroid with
    # some element in the data
    for i in range(K):
        centroids[i, :] = X[idx[i], :]

    return centroids


def main():
    # Load the data (MATLAB Data)
    data = loadmat("Data/ex7data2.mat")
    X = data['X']
    # print(X.shape)
    # m = 300, n = 2

    # Initialize the three centeroids
    initial_centroids = initialize_centroids(X, 3)
    print(initial_centroids)

    # Find closest centroid for each point in the data
    # idx = find_closest_centroids(X, initial_centroids)
    # print(idx)

    # Move the cenroids
    # new_centroids = move_centroids(X, idx, 3)
    # print(new_centroids)


    # Iterations
    max_iters = 5

    for i in range(3):
        # Run k-means and get the indecies and centroids
        idx, centroids = run_k_means(X, initial_centroids, max_iters)
    
        print("IDX: ", idx)
        print("CENTROIDS: ", centroids)

        # Plot the data and the clusters 
        plot_clusters(X, idx, centroids)



if __name__ == '__main__':
    plt.style.use('ggplot')
    main()
