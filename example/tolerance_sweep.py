"""Measure compression-only and full-pipeline errors independently."""
import numpy as np
from fABBA import fABBA, inverse_compress


def main():
    x = np.sin(np.linspace(0, 12, 300)) + 0.1 * np.cos(np.linspace(0, 37, 300))
    print("tol       alpha     pieces  alphabet  polygon RMSE  symbolic RMSE")
    for tol, alpha in [(0.1, 0.5), (0.01, 0.1), (0.001, 0.01), (0.0, 0.0)]:
        model = fABBA(tol=tol, alpha=alpha, verbose=0)
        symbols = model.fit_transform(x)
        polygon = np.asarray(inverse_compress(model.pieces_, start=x[0]))
        decoded = np.asarray(model.inverse_transform(symbols, start=x[0]))
        polygon_rmse = np.sqrt(np.mean((polygon - x) ** 2))
        symbolic_rmse = np.sqrt(np.mean((decoded - x) ** 2))
        print(f"{tol:<9g} {alpha:<9g} {len(symbols):6d} {len(model.parameters.centers):9d} "
              f"{polygon_rmse:13.6g} {symbolic_rmse:14.6g}")
        if tol == 0:
            np.testing.assert_allclose(decoded, x, atol=1e-12)
    # Greedy segmentation and clustering can change memberships abruptly.
    # Do not assume monotonically decreasing end-to-end RMSE at every step.


if __name__ == "__main__":
    main()
