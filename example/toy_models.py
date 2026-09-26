"""Five reproducible signals: run `python example/toy_models.py` after installation."""
import numpy as np
from fABBA import fABBA


def main():
    t = np.linspace(0, 4 * np.pi, 240)
    rng = np.random.default_rng(42)
    signals = {
        "constant": np.full(t.size, 3.0),
        "linear trend": 2 + 0.2 * t,
        "periodic": np.sin(t),
        "step": np.where(t < 2 * np.pi, 0.0, 1.0),
        "noisy periodic": np.sin(t) + 0.05 * rng.normal(size=t.size),
    }
    print(f"{'signal':18s} {'samples':>7s} {'symbols':>7s} {'alphabet':>8s} {'RMSE':>10s}")
    for name, x in signals.items():
        model = fABBA(tol=0.001, alpha=0.05, verbose=0)
        symbols = model.fit_transform(x)
        y = np.asarray(model.inverse_transform(symbols, start=x[0]))
        assert y.shape == x.shape
        rmse = np.sqrt(np.mean((x - y) ** 2))
        print(f"{name:18s} {len(x):7d} {len(symbols):7d} {len(model.parameters.centers):8d} {rmse:10.5f}")
        print("  symbols:", symbols[:80])
        # Rows of centers are [segment length, increment], in original units.
        assert np.isfinite(model.parameters.centers).all()


if __name__ == "__main__":
    main()
