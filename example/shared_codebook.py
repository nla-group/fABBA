"""Toy classification using a JABBA codebook fitted only on training signals."""
import numpy as np
from fABBA import JABBA


def main():
    t = np.linspace(0, 4 * np.pi, 180)
    train = np.asarray([np.sin(t), np.sin(t + 0.15), np.sin(3*t), np.sin(3*t + 0.15)])
    labels = np.array([0, 0, 1, 1])  # low versus high frequency
    test = np.asarray([np.sin(t + 0.08), np.sin(3*t + 0.08)])
    model = JABBA(tol=0.001, alpha=0.1, verbose=0, random_state=42)
    train_symbols = model.fit_transform(train, n_jobs=1)
    centers_before = model.parameters.centers.copy()
    test_symbols, test_starts = model.transform(test, n_jobs=1)
    np.testing.assert_array_equal(centers_before, model.parameters.centers)
    alphabet = list(model.parameters.alphabets)

    def histogram(symbols):
        counts = np.array([list(symbols).count(symbol) for symbol in alphabet], dtype=float)
        return counts / counts.sum()

    features = np.asarray([histogram(s) for s in train_symbols])
    queries = np.asarray([histogram(s) for s in test_symbols])
    distances = np.linalg.norm(queries[:, None, :] - features[None, :, :], axis=2)
    predictions = labels[distances.argmin(axis=1)]
    reconstructed = model.inverse_transform(test_symbols, start_set=test_starts, n_jobs=1)
    print("Predicted frequency classes:", predictions.tolist())
    print("Expected toy classes:      ", [0, 1])
    print("Shared alphabet size:", len(alphabet))
    print("Decoded lengths:", [len(row) for row in reconstructed])
    assert np.isfinite(features).all()
    # Symbol histograms ignore ordering. This illustrates an API workflow,
    # not a validated classifier or a claim about real-world performance.


if __name__ == "__main__":
    main()
