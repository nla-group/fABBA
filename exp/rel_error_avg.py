from aeon.datasets import load_classification
from normalization import NormalizationPipeline
from fABBA import JABBA
import numpy as np
import csv
from pathlib import Path
import matplotlib.pyplot as plt


UEA_30 = [
    "ArticularyWordRecognition",
    "AtrialFibrillation",
    "BasicMotions",
    "CharacterTrajectories",
    "Cricket",
    "DuckDuckGeese",
    "EigenWorms",
    "Epilepsy",
    "EthanolConcentration",
    "ERing",
    "FaceDetection",
    "FingerMovements",
    "HandMovementDirection",
    "Handwriting",
    "Heartbeat",
    "InsectWingbeat",
    "JapaneseVowels",
    "Libras",
    "LSST",
    "MotorImagery",
    "NATOPS",
    "PenDigits",
    "PEMS-SF",
    "Phoneme",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "SpokenArabicDigits",
    "StandWalkJump",
    "UWaveGestureLibrary",
]


def dataset_rel_recon_error(X, X_rec) -> float:
    """Sum_i ||Xi - X̂i||_F / Sum_i ||Xi||_F"""
    num, den = 0.0, 0.0
    for i in range(len(X)):
        num += np.linalg.norm(X[i] - X_rec[i], ord="fro")
        den += np.linalg.norm(X[i], ord="fro")
    return num / den if den != 0 else np.nan


def jabba_reconstruct(X, *, last_dim: bool, tol: float, k: int, verbose: int = 0):
    """
    Fit JABBA on X, return:
      - X_rec: reconstructed X (np.ndarray or list-like indexed by sample)
      - err: dataset-level relative reconstruction error
    """
    jabba = JABBA(
        tol=tol,
        init="kmeans",
        k=k,
        verbose=verbose,
        last_dim=last_dim
    )
    symbols = jabba.fit_transform(X)
    X_rec = jabba.inverse_transform(symbols)
    err = dataset_rel_recon_error(X, X_rec)
    return X_rec, err


def setup_matplotlib():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2,
        "lines.markersize": 5,
        "font.family": "sans-serif",
    })


def plot_sample_dims_recon(
    *,
    dataset_name: str,
    Xi: np.ndarray,
    Xi_rec_proposed: np.ndarray,
    Xi_rec_baseline: np.ndarray,
    out_pdf: Path,
    out_jpg: Path,
):
    C, T = Xi.shape
    dims = [0] if C < 2 else [0, 1]
    t = np.arange(T)

    nrows = len(dims)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(9.5, 3.2 * nrows),
        sharex=True,
    )
    if nrows == 1:
        axes = [axes]

    for ax, d in zip(axes, dims):
        ax.plot(
            t,
            Xi[d],
            label="Input (normalized)",
            color="black",
            linestyle="-",
            marker="o",
            markevery=max(T // 30, 1),
            alpha=0.9,
            zorder=3,
        )

        ax.plot(
            t,
            Xi_rec_proposed[d],
            label="Reconstruction (proposed, last_dim=True)",
            color="#1f77b4",
            linestyle="--",
            marker="s",
            markevery=max(T // 30, 1),
            alpha=0.85,
            zorder=2,
        )

        ax.plot(
            t,
            Xi_rec_baseline[d],
            label="Reconstruction (baseline, last_dim=False)",
            color="#d62728",
            linestyle=":",
            marker="^",
            markevery=max(T // 30, 1),
            alpha=0.85,
            zorder=1,
        )

        ax.set_ylabel(f"dim {d}", fontsize=15)

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=13,
            length=6,
            width=1.2,
        )

        ax.grid(True, linestyle="--", alpha=0.3)

    axes[-1].set_xlabel("time index", fontsize=15)

    axes[0].legend(
        loc="best",
        fontsize=13,
        frameon=True,
        framealpha=0.1,
        fancybox=True,
    )

    fig.suptitle(dataset_name, fontsize=16, y=0.92)

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_jpg, bbox_inches="tight", dpi=300)
    plt.close(fig)



setup_matplotlib()

TOL = 0.005
K_MAX = 800

OUT_DIR = Path("results_summary")
OUT_DIR.mkdir(exist_ok=True)

PLOT_DIR = OUT_DIR / "reconstruction_plots"
PLOT_DIR.mkdir(exist_ok=True)

csv_path = OUT_DIR / "jabba_lastdim_comparison.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "dataset",
        "cases",
        "channels",
        "length",
        "classes",
        "k",
        "tol",
        "err_proposed",
        "err_baseline",
        "relative_improvement"
    ])

for name in UEA_30:
    print(f"\n=== {name} ===")
    try:
        X, y = load_classification(name)

        # normalization (your current choice)
        pipe = NormalizationPipeline(mode="minmax_per_channel")
        X = pipe.fit_transform(X)  # keep as list of (C, T)

        n_cases = len(X)
        n_channels, series_length = X[0].shape
        n_classes = len(np.unique(y))

        print(f"cases={n_cases}, channels={n_channels}, length={series_length}, classes={n_classes}")

        # choose k (robust min/max)
        k = min(K_MAX, series_length)
        k = max(10, k)
        print(f"Using k={k}, tol={TOL}")

        # proposed reconstruction + error
        X_rec_proposed, err_joint = jabba_reconstruct(
            X, last_dim=True, tol=TOL, k=k, verbose=0
        )
        print(f"Structured (last-dim preserved): {err_joint:.4f}")

        # baseline reconstruction + error
        X_rec_baseline, err_sep = jabba_reconstruct(
            X, last_dim=False, tol=TOL, k=k, verbose=0
        )
        print(f"Flattened (last-dim merged){err_sep:.4f}")

        # improvement (>0 means proposed better)
        improvement = (err_sep - err_joint) / err_sep if (err_sep != 0 and not np.isnan(err_sep)) else np.nan
        print(f"Relative improvement: {improvement * 100:.2f}%")

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                name,
                n_cases,
                n_channels,
                series_length,
                n_classes,
                k,
                TOL,
                err_joint,
                err_sep,
                improvement
            ])

        # Visualization: sample 0, first 2 dims
        Xi = X[0]  # (C, T)

        # X_rec_* might be np.ndarray (N, C, T) or list-like; handle both
        Xi_rec_prop = X_rec_proposed[0] if not isinstance(X_rec_proposed, list) else X_rec_proposed[0]
        Xi_rec_base = X_rec_baseline[0] if not isinstance(X_rec_baseline, list) else X_rec_baseline[0]

        out_pdf = PLOT_DIR / f"{name}_sample0_dims01_reconstruction.pdf"
        out_jpg = PLOT_DIR / f"{name}_sample0_dims01_reconstruction.jpg"

        plot_sample_dims_recon(
            dataset_name=name,
            Xi=Xi,
            Xi_rec_proposed=Xi_rec_prop,
            Xi_rec_baseline=Xi_rec_base,
            out_pdf=out_pdf,
            out_jpg=out_jpg,
        )

        print(f"Saved plot: {out_pdf.name} and {out_jpg.name}")

    except Exception as e:
        print("FAILED:", repr(e))
