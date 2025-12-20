import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from aeon.datasets import load_classification
from normalization import NormalizationPipeline
from fABBA import JABBA
from pathlib import Path
from tqdm import tqdm


# =======================
# Configuration
# =======================

DATASETS = [
    "AtrialFibrillation",
    "CharacterTrajectories",
    "Epilepsy",
    "StandWalkJump",
]

K_LIST = [100, 200, 300, 400, 500, 600]
FIXED_TOL = 0.005

OUT_DIR = Path("results_k_error")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =======================
# Matplotlib (NeurIPS style)
# =======================

def setup_matplotlib():
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
    })


setup_matplotlib()


COLOR_PROPOSED = "#0072B2"
COLOR_BASELINE = "#D55E00"


# =======================
# Utils
# =======================

def dataset_rel_recon_error(X, X_rec):
    num, den = 0.0, 0.0
    for i in range(len(X)):
        num += np.linalg.norm(X[i] - X_rec[i], ord="fro")
        den += np.linalg.norm(X[i], ord="fro")
    return num / den


def run_jabba(X, *, last_dim, tol, k):
    jabba = JABBA(
        tol=tol,
        init="kmeans",
        k=k,
        verbose=0,
        last_dim=last_dim,
    )
    symbols = jabba.fit_transform(X)
    X_rec = jabba.inverse_transform(symbols)
    return dataset_rel_recon_error(X, X_rec)


# =======================
# Run experiments
# =======================

results = {}

for name in tqdm(DATASETS):
    print(f"Running {name}...")
    X, y = load_classification(name)
    X = NormalizationPipeline(
        mode="minmax_per_channel"
    ).fit_transform(X)

    _, series_length = X[0].shape

    err_joint, err_sep = [], []

    for k in K_LIST:
        k_eff = min(series_length, max(10, k))

        err_joint.append(
            run_jabba(
                X, last_dim=True, tol=FIXED_TOL, k=k_eff
            )
        )
        err_sep.append(
            run_jabba(
                X, last_dim=False, tol=FIXED_TOL, k=k_eff
            )
        )

    results[name] = (err_joint, err_sep)


# =======================
# Compute global y-limits
# =======================

all_errors = []
for v in results.values():
    all_errors.extend(v[0])
    all_errors.extend(v[1])

y_min, y_max = min(all_errors), max(all_errors)
margin = 0.05 * (y_max - y_min)
y_lim = (y_min - margin, y_max + margin)


# =======================
# Plot (multi-panel)
# =======================

fig, axes = plt.subplots(
    2, 2,
    figsize=(7.2, 6.6),
    sharex=True,
    sharey=True,
)

axes = axes.flatten()

for ax, (name, (err_joint, err_sep)) in zip(
    axes, results.items()
):
    ax.plot(
        K_LIST,
        err_joint,
        marker="o",
        color=COLOR_PROPOSED,
        label="Structured (last-dim preserved)",
        markerfacecolor="white",
        markeredgewidth=1.5,
    )
    ax.plot(
        K_LIST,
        err_sep,
        marker="s",
        color=COLOR_BASELINE,
        label="Flattened (last-dim merged)",
        markerfacecolor="white",
        markeredgewidth=1.5,
    )

    ax.set_title(name)
    ax.set_ylim(*y_lim)
    ax.grid(True, axis="y")
    ax.yaxis.set_major_formatter(
        FormatStrFormatter("%.3f")
    )

for ax in axes[2:]:
    ax.set_xlabel("k (dictionary size)")
for ax in axes[::2]:
    ax.set_ylabel("Relative reconstruction error")

# Global legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 1.0),
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig(
    OUT_DIR / "k_error_all_datasets.pdf",
    bbox_inches="tight",
)
plt.savefig(
    OUT_DIR / "k_error_all_datasets.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()
