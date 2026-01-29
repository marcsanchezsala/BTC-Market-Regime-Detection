import numpy as np
import ot

class WKMeans:
    def __init__(self, k, p=1, tolerance=1e-4, max_iter=100, seed=42):
        """
        Wasserstein K-means algorithm.

        Parameters:
        - k: Number of clusters
        - p: Order of the Wasserstein distance
        - tolerance: Convergence threshold
        - max_iter: Maximum number of iterations
        """
        self.k = k
        self.p = p
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.seed = seed
        self.centroids = None

    def wasserstein_distance(self, mu1, mu2):
        """Compute the p-Wasserstein distance between two empirical distributions."""
        n = len(mu1)
        a = np.ones(n) / n
        M = ot.dist(mu1.reshape(-1, 1), mu2.reshape(-1, 1), metric="minkowski", p=self.p)
        return ot.emd2(a, a, M)

    def wasserstein_barycenter(self, cluster_samples):
        """Compute the Wasserstein barycenter (median of sorted distributions)."""
        sorted_samples = np.sort(cluster_samples, axis=0)
        return np.median(sorted_samples, axis=0)

    def fit(self, samples, inertia = False):
        """
        Fit the WK-means clustering algorithm.

        Parameters:
        - samples: List of empirical distributions (numpy arrays)
        """
        # Initialize centroids by randomly selecting k samples
        np.random.seed(self.seed)
        self.centroids = [samples[i] for i in np.random.choice(len(samples), self.k, replace=False)]

        for _ in range(self.max_iter):
            clusters = {i: [] for i in range(self.k)}

            # Assign each sample to the closest centroid
            for sample in samples:
                distances = [self.wasserstein_distance(sample, centroid) for centroid in self.centroids]
                closest_cluster = np.argmin(distances)
                clusters[closest_cluster].append(sample)

            # Update centroids as Wasserstein barycenters
            new_centroids = []
            for i in range(self.k):
                if clusters[i]:
                    new_centroids.append(self.wasserstein_barycenter(np.array(clusters[i])))
                else:
                    new_centroids.append(self.centroids[i])  # Keep previous centroid if no samples assigned

            # Compute loss function (sum of Wasserstein distances)
            loss = sum(self.wasserstein_distance(self.centroids[i], new_centroids[i]) for i in range(self.k))

            # Check convergence
            if loss < self.tolerance:
                break

            self.centroids = new_centroids
        
        if inertia:
            self.inertia_ = 0
            for sample in samples:
                dist = min([self.wasserstein_distance(sample, c) for c in self.centroids])
                self.inertia_ += dist**2

    def predict(self, samples):
        """
        Predict the cluster for each sample.

        Parameters:
        - samples: List of empirical distributions (numpy arrays)

        Returns:
        - List of cluster indices
        """
        return [np.argmin([self.wasserstein_distance(sample, centroid) for centroid in self.centroids]) for sample in samples]