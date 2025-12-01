import faiss
import numpy as np
import time

def faiss_kmeans_cluster(data_vectors: np.ndarray, 
                         n_clusters: int, 
                         n_iterations: int = 25, n_redo: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs K-means clustering on a set of vectors using the FAISS library.

    FAISS is optimized for high-dimensional and large-scale vector operations.
    Note: FAISS expects vectors to be in float32 format.


    Parameters
    ----------
    data_vectors (np.ndarray): 
        The input data matrix (N x D), where N is 
        the number of vectors and D is the dimension.
        Must be convertible to float32.
    
    n_clusters (int): 
        The desired number of clusters (K).

    n_iterations (int): 
        The maximum number of K-means iterations. (default 25)
        
    n_redo (int): 
        The number of times to run K-means with different random
        initializations, keeping the best result. (default 1)

        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]: A tuple containing:
        - centroids (np.ndarray): The final cluster centers (K x D).
        - labels (np.ndarray): The cluster label assigned to each input vector (N,).

    Raises:
        ValueError: If the input data is not 2-dimensional.
    """
    if data_vectors.ndim != 2:
        raise ValueError("Input 'data_vectors' must be a 2D numpy array (N x D).")

    # 1. FAISS Requirement: Convert data to float32
    d = data_vectors.shape[1] # Dimension of the vectors
    data_vectors_f32 = data_vectors.astype('float32')

    # 2. Initialize the Kmeans object
    # Kmeans constructor requires dimension (d) and number of centroids (n_clusters)
    kmeans = faiss.Kmeans(
        d, 
        n_clusters, 
        niter=n_iterations, 
        nredo=n_redo, 
        verbose=True, 
        gpu=False # Set to True if a GPU is available and you installed faiss-gpu
    )

    # 3. Train the K-means model
    print(f"--- Starting FAISS K-means training (K={n_clusters}, D={d}) ---")
    start_time = time.time()
    kmeans.train(data_vectors_f32)
    end_time = time.time()
    print(f"--- Training finished in {end_time - start_time:.2f} seconds ---")

    # 4. Extract Cluster Centers (Centroids)
    # The centroids are stored in the 'centroids' attribute of the Kmeans object.
    centroids = kmeans.centroids

    # 5. Get Cluster Labels for all input vectors
    # We use the internal index object created during training to search for the
    # nearest centroid for every input vector. k=1 means we only want the single
    # nearest neighbor (the cluster center).
    # D: Distances (squared L2) to the nearest centroids
    # I: Indices of the nearest centroids (the cluster labels)
    D, I = kmeans.index.search(data_vectors_f32, 1)

    # I is typically a 2D array of shape (N, 1), so we flatten it to get 1D labels.
    labels = I.flatten()

    return centroids, labels