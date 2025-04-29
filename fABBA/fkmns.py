import numpy as np
from scipy.linalg import get_blas_funcs
from sklearn.cluster import KMeans
from sklearn.cluster import kmeans_plusplus

    
### Deprecated
def euclid_skip(xxt, xv, v):
    ed2 = xxt + np.inner(v,v).ravel() - xv
    return ed2

    
### Deprecated
def calculate_shortest_distance_refine1(data, centers):
    distance = np.empty([data.shape[0], centers.shape[0]])
    xxv = np.einsum('ij,ij->i',data, data) 
    xv = 2* np.inner(data, centers) #data.dot(centers.T) # BLAS LEVEL 3
    for i in range(centers.shape[0]):
        distance[:, i] = euclid_skip(xxv, xv[:,i], centers[i]) # LA.norm(data - centers[i], axis=1)
        
    return np.min(distance, axis=1)

    
### Deprecated
def calculate_shortest_distance_refine2(data, centers): # involve additional memory copy, is slow for high dimensional data
    distance = np.empty([data.shape[0], centers.shape[0]])
    xxv = np.einsum('ij,ij->i',data, data) 
    gemm = get_blas_funcs("gemm", [data, centers.T])
    xv = 2*gemm(1, data, centers.T)
    for i in range(centers.shape[0]):
        distance[:, i] = euclid_skip(xxv, xv[:,i], centers[i]) # LA.norm(data - centers[i], axis=1)
        
    return np.min(distance, axis=1)

    
### Deprecated
def calculate_shortest_distance_label(data, centers):
    distance = np.empty([data.shape[0], centers.shape[0]])
    xxv = np.einsum('ij,ij->i',data, data) 
    xv = 2*data.dot(centers.T) # BLAS LEVEL 3
    for i in range(centers.shape[0]):
        distance[:, i] = euclid_skip(xxv, xv[:,i], centers[i]) # LA.norm(data - centers[i], axis=1)
        
    return np.argmin(distance, axis=1)


### Deprecated
class kmeans:
    def __init__(self, n_clusters=1, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.labels_ = None
        self.record_iters = None # the iterations
        
    def fit_predict(self, X, init_centers):
        return self.fit(X, init_centers).labels_
        
    def fit(self, X, init_centers):
        self.centers = init_centers
        self.record_iters = 0
        prev_centers = None
        while np.not_equal(self.centers, prev_centers).any() and self.record_iters < self.max_iter:
            self.labels_ = calculate_shortest_distance_label(X, self.centers)

            prev_centers = self.centers.copy()
            for i in range(self.n_clusters):
                self.centers[i] = np.mean(X[self.labels_ == i], axis=0)
                
            for i, center in enumerate(self.centers):
                if np.isnan(center).any():  # Catch any np.nans, resulting from a centroid having no points
                    self.centers[i] = prev_centers[i]
                    
            self.record_iters += 1
            
        return self
            
    def predict(self, X):
        return calculate_shortest_distance_label(X, self.centers)
    
    

    
### Deprecated
class kmeanspp:
    def __init__(self, n_clusters=1, max_iter=300, random_state=4022, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.labels_ = None
        self.record_iters = None # the iterations
        self.random_state = random_state
        self.mu = None
        self.tol = tol
        
    def fit_predict(self, X):
        return self.fit(X).labels_
        
    def fit(self, X):
        
        self.mu = X.mean(axis=0)
        # The copy was already done above
        X -= self.mu

        self.centers, _ = kmeans_plusplus(X, self.n_clusters, random_state=self.random_state)
        self.record_iters = 0
        self.prev_centers = np.zeros((self.n_clusters, X.shape[1]))
        while self.record_iters < self.max_iter:
            self.labels_ = calculate_shortest_distance_label(X, self.centers)

            self.prev_centers = self.centers.copy()
            for i in range(self.n_clusters):
                self.centers[i] = np.mean(X[self.labels_ == i], axis=0)
                
            for i, center in enumerate(self.centers):
                if np.isnan(center).any():  # Catch any np.nans, resulting from a centroid having no points
                    self.centers[i] = self.prev_centers[i]
                    
            self.record_iters += 1
           
            if np.linalg.norm(self.centers - self.prev_centers, 'fro')<=self.tol:
                break
            
        return self
    
    def predict(self, X):
        if self.mu is None:
            raise ValueError('Please fit before predict.')
        return calculate_shortest_distance_label(X, self.centers)
    
    

def uniform_sample(X, size=100, random_state=42):
    """
    initialized the centroids with uniform initialization
    
    inputs:
        X - numpy array of data points having shape (n_samples, n_dim)
        size - number of clusters
    """
    np.random.seed(random_state)
    subsampleID = np.random.choice(X.shape[0], size=size, replace=False)
    return subsampleID



def calculate_cluster_centers(data, labels):
    """Calculate the mean centers of clusters from given data."""
    classes = np.unique(labels)
    centers = np.zeros((len(classes), data.shape[1]))
    for c in classes:
        centers[c] = np.mean(data[labels==c,:], axis=0)
    return centers


class sampledKMeansInter(KMeans):
    def __init__(self, 
                 n_clusters=1,
                 r=0.5,
                 init='k-means++',
                 max_iter=300, 
                 random_state=42, 
                 tol=1e-4):
        
        super().__init__(n_clusters=n_clusters, 
                         init=init,
                         max_iter=max_iter,
                         random_state=random_state, 
                         n_init=1, # consistent to the default setting of sklearn k-means
                         tol=tol)
        self.r = r
        self.size = None
        
    def sampled_fit_predict(self, X):
        return self.sampled_fit(X).labels_
        
    def sampled_fit(self, X):
        self.size = int(self.r*X.shape[0])
        if self.n_clusters >= self.size:
            self.size = self.n_clusters
            
        self.core_pts = uniform_sample(X, self.size, self.random_state)
        labels_ = np.zeros(X.shape[0], dtype=int)
        index = np.zeros(X.shape[0], dtype=bool)
        index[self.core_pts] = True
        labels_[index] = self.fit_predict(X[index])
        inverse_labels = ~index 
        if np.any(inverse_labels):
            labels_[inverse_labels] = self.predict(X[inverse_labels])
        self.labels_ = labels_
        
        self.cluster_centers_ = calculate_cluster_centers(X, self.labels_)
        return self
            
    def sampled_predict(self, X):
        if self.mu is None:
            raise ValueError('Please fit before predict.')
        return self.predict(X)
    
