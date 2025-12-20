import numpy as np

class NormalizationPipeline:
    """
    Unified normalization + inverse pipeline for multivariate time series.

    X format:
        list of ndarray, each shape (channels, time)
    """

    def __init__(self, mode: str, eps: float = 1e-12, auto_stack: bool=True):
        """
        mode options:
            - "z_per_channel"
            - "z_per_series"
            - "z_dataset"
            - "minmax_per_channel"
            - "minmax_per_series"
            - "minmax_dataset"
        """
        self.mode = mode
        self.eps = eps
        self.params = None
        self.auto_stack = auto_stack

    def fit_transform(self, X):
        if self.mode == "z_per_channel":
            X_rec = self._z_per_channel(X)

        elif self.mode == "z_per_series":
            X_rec = self._z_per_series(X)
  
        elif self.mode == "z_dataset":
            X_rec = self._z_dataset(X)

        elif self.mode == "minmax_per_channel":
            X_rec = self._minmax_per_channel(X)

        elif self.mode == "minmax_per_series":
            X_rec = self._minmax_per_series(X)

        elif self.mode == "minmax_dataset":
            X_rec = self._minmax_dataset(X)

        else:
            raise ValueError(f"Unknown normalization mode: {self.mode}")

        if self.auto_stack:
            return np.stack(X_rec)
        else:
            return X_rec

    def inverse_transform(self, Xn):
        if self.params is None:
            raise RuntimeError("inverse_transform called before fit_transform")

        if self.mode in ("z_per_channel", "z_per_series", "z_dataset"):
            return self._z_inverse(Xn)
        elif self.mode in ("minmax_per_channel", "minmax_per_series", "minmax_dataset"):
            return self._minmax_inverse(Xn)
        else:
            raise ValueError(f"Unknown normalization mode: {self.mode}")

    # Z-normalization
    def _z_per_channel(self, X):
        Xn, params = [], []
        for Xi in X:
            mu = Xi.mean(axis=1, keepdims=True)
            std = Xi.std(axis=1, keepdims=True) + self.eps
            Xn.append((Xi - mu) / std)
            params.append((mu, std))
        self.params = params
        return Xn

    def _z_per_series(self, X):
        Xn, params = [], []
        for Xi in X:
            mu = Xi.mean()
            std = Xi.std() + self.eps
            Xn.append((Xi - mu) / std)
            params.append((mu, std))
        self.params = params
        return Xn

    def _z_dataset(self, X):
        all_values = np.concatenate([Xi.reshape(-1) for Xi in X])
        mu = all_values.mean()
        std = all_values.std() + self.eps
        self.params = (mu, std)
        return [(Xi - mu) / std for Xi in X]

    def _z_inverse(self, Xn):
        X = []
        if isinstance(self.params, list):  # per-sample
            for Xi_n, (mu, std) in zip(Xn, self.params):
                X.append(Xi_n * std + mu)
        else:  # dataset-level
            mu, std = self.params
            X = [Xi_n * std + mu for Xi_n in Xn]
        return X

    # Min–Max normalization
    def _minmax_per_channel(self, X):
        Xn, params = [], []
        for Xi in X:
            minv = Xi.min(axis=1, keepdims=True)
            maxv = Xi.max(axis=1, keepdims=True)
            scale = maxv - minv + self.eps
            Xn.append((Xi - minv) / scale)
            params.append((minv, scale))
        self.params = params
        return Xn

    def _minmax_per_series(self, X):
        Xn, params = [], []
        for Xi in X:
            minv = Xi.min()
            maxv = Xi.max()
            scale = maxv - minv + self.eps
            Xn.append((Xi - minv) / scale)
            params.append((minv, scale))
        self.params = params
        return Xn

    def _minmax_dataset(self, X):
        all_values = np.concatenate([Xi.reshape(-1) for Xi in X])
        minv = all_values.min()
        maxv = all_values.max()
        scale = maxv - minv + self.eps
        self.params = (minv, scale)
        return [(Xi - minv) / scale for Xi in X]

    def _minmax_inverse(self, Xn):
        X = []
        if isinstance(self.params, list):  # per-sample
            for Xi_n, (minv, scale) in zip(Xn, self.params):
                X.append(Xi_n * scale + minv)
        else:  # dataset-level
            minv, scale = self.params
            X = [Xi_n * scale + minv for Xi_n in Xn]
        return X


