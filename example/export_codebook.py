"""Export symbols + starting value + codebook to JSON, then decode without fitting."""
import json
import tempfile
from pathlib import Path
import numpy as np
from fABBA import fABBA, Model, __version__


def main():
    x = 5 + np.sin(np.linspace(0, 4 * np.pi, 200))
    model = fABBA(tol=0.01, alpha=0.1, verbose=0)
    symbols = model.fit_transform(x)
    payload = {
        "fabba_version": __version__,
        "config": {"tol": model.tol, "alpha": model.alpha, "scl": model.scl,
                   "sorting": model.sorting, "max_len": model.max_len},
        "start": float(x[0]), "n_samples": len(x), "symbols": symbols,
        "codebook": model.parameters.to_dict(),
    }
    # Replace this temporary path with Path("signal.json") to retain the export.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "signal.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        restored = json.loads(path.read_text(encoding="utf-8"))
        parameters = Model.from_dict(restored["codebook"])
        decoder = fABBA(verbose=0)
        y = np.asarray(decoder.inverse_transform(restored["symbols"],
                       start=restored["start"], parameters=parameters))
        np.testing.assert_allclose(y, model.inverse_transform(symbols, x[0]))
        assert len(y) == restored["n_samples"]
        print("JSON roundtrip verified; alphabet size:", len(parameters.alphabets))
        print("Reconstruction RMSE:", np.sqrt(np.mean((x - y) ** 2)))
        model.print_parameters()


if __name__ == "__main__":
    main()
