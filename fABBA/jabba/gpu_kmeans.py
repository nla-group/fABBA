import torch
import warnings

def kmeans_fp32(X, k, max_iter=100, tol=1e-4):
    """
    Performs K-means clustering with D squared initialization in float32 precision.


    Parameters
    ----------
    X (torch.Tensor): 
        The input data tensor.

    k (int): 
        The number of clusters.

    max_iter (int): 
        The maximum number of K-means iterations.

    tol (float): 
    The convergence tolerance.


    Returns
    ----------
    tuple: (centroids, labels) where centroids are the final cluster centers
        and labels are the cluster assignments for each data point.
    """
    # --- Configuration for FP32 ---
    DTYPE = torch.float32 
    TOL_FP32 = float(tol) # Ensure tolerance is a standard float

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if device.type == 'cpu':
        warnings.warn("CUDA not found, run in CPU.")
    
    # 1. Data Preparation (Ensure everything is FP32)
    X = X.to(device).to(DTYPE) # Data moved to device and converted to FP32
    N, D = X.shape
    
    # Normalize data
    X_mean = X.mean(dim=0)
    X_std = X.std(dim=0)
    X_std[X_std == 0] = 1.0
    X = (X - X_mean) / X_std
    X_sq_norm = torch.sum(X ** 2, dim=1, keepdim=True)

    # 2. K-means++ Initialization (Ensure all tensors are FP32)
    centroids = torch.empty((k, D), device=device, dtype=DTYPE)
    rand_idx = torch.randint(0, N, (1,)).item()
    centroids[0] = X[rand_idx]
    current_dist_sq = torch.full((N, 1), float('inf'), device=device, dtype=DTYPE) # Use DTYPE
    
    for i in range(1, k):
        latest_center = centroids[i-1].view(1, D)
        c_sq_norm = torch.sum(latest_center ** 2)
        dot_prod = torch.mm(X, latest_center.T)
        # Distance calculation: ||x - c||^2 = ||x||^2 - 2 * x^T * c + ||c||^2
        new_dists = X_sq_norm - 2 * dot_prod + c_sq_norm
        # Clamp to ensure non-negative distances due to potential floating-point errors
        new_dists = torch.clamp(new_dists, min=0.0) 
        current_dist_sq = torch.minimum(current_dist_sq, new_dists)
        
        # Select next center based on distances squared (weighted probability)
        # Sum must be non-zero for multinomial to work.
        sum_dist_sq = torch.sum(current_dist_sq)
        if sum_dist_sq == 0:
            next_center_idx = torch.randint(0, N, (1,)).item() # Fallback to random
        else:
            probs = current_dist_sq.squeeze() / sum_dist_sq
            next_center_idx = torch.multinomial(probs, 1).item()

        centroids[i] = X[next_center_idx]

    # 3. K-means Iteration (Ensure all tensors are FP32)
    # Tensors for accumulating cluster sums and counts
    acc = torch.zeros((k, D), dtype=DTYPE, device=device)
    counts = torch.zeros(k, dtype=DTYPE, device=device)
    labels = torch.empty(N, dtype=torch.long, device=device)
    ones = None # will be initialized with DTYPE

    for _ in range(max_iter):
        prev = centroids.clone()
        
        # Calculate distances: ||x - c||^2 = ||x||^2 - 2 * x^T * c + ||c||^2
        # Note: torch.sum(centroids**2, dim=1) results in a 1D tensor of shape (k,)
        # It's broadcasted across the rows of X.
        dists = X_sq_norm - 2 * torch.mm(X, centroids.T) + torch.sum(centroids**2, dim=1)
        labels = torch.argmin(dists, dim=1)

        # Recalculate centroids
        acc.zero_()
        counts.zero_()
        
        # Optimization: only re-create 'ones' if N changes or it hasn't been created
        if ones is None or ones.shape[0] != N:
            # Use DTYPE for 'ones' to match 'counts' and 'acc'
            ones = torch.ones_like(labels, dtype=DTYPE) 

        # Sum of points in each cluster
        acc.index_add_(0, labels, X)
        # Count of points in each cluster
        counts.index_add_(0, labels, ones)

        # Update centroids: centroids = acc / counts
        valid = counts > 0
        # Only update centroids for clusters that have points
        centroids = torch.where(valid.view(k, 1), acc / counts.view(k, 1), centroids)

        # Check for convergence
        # Convergence criterion: ||centroids - prev|| <= tol * ||prev||
        # Ensure comparison is done with FP32 values
        if torch.norm(centroids - prev) <= TOL_FP32 * torch.norm(prev):
            break

    return centroids, labels


if __name__ == "__main__":
    # --- Example Usage (Optional: run this to test) ---
    N_SAMPLES = 100000
    N_FEATURES = 10
    N_CLUSTERS = 3

    # # Create dummy data - it will be converted to FP32 inside the function
    X_data = torch.randn(N_SAMPLES, N_FEATURES)

    print("\nStarting K-means FP32...")
    final_centroids, final_labels = kmeans_fp32(X_data, k=N_CLUSTERS)
    print("K-means FP32 completed.")
    print(f"Final Centroids dtype: {final_centroids.dtype}")
    print(f"Final Labels dtype: {final_labels.dtype}")