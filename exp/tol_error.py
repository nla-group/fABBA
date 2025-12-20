import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter, ScalarFormatter

from normalization import NormalizationPipeline
from aeon.datasets import load_classification
from fABBA import JABBA
from pathlib import Path
from tqdm import tqdm


DATASETS = [
    "AtrialFibrillation",
    "CharacterTrajectories",
    "Epilepsy",
    "StandWalkJump",
]

TOL_LIST = np.array([1, 3, 5, 7, 9]) * 1e-3

K_MAX = 800

OUT_DIR = Path("results_tol_error")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLOR_PROPOSED = "#0072B2"
COLOR_BASELINE = "#D55E00"


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


results = {}

for name in tqdm(DATASETS):
    print(f"Running {name}...")

    X, y = load_classification(name)
    X = NormalizationPipeline(
        mode="minmax_per_channel"
    ).fit_transform(X)

    _, series_length = X[0].shape
    FIXED_K = min(series_length, max(10, K_MAX))

    err_joint, err_sep = [], []

    for tol in TOL_LIST:
        err_joint.append(
            run_jabba(
                X,
                last_dim=True,
                tol=tol,
                k=FIXED_K,
            )
        )
        err_sep.append(
            run_jabba(
                X,
                last_dim=False,
                tol=tol,
                k=FIXED_K,
            )
        )

    results[name] = (err_joint, err_sep)


all_errors = []
for err_joint, err_sep in results.values():
    all_errors.extend(err_joint)
    all_errors.extend(err_sep)

y_min, y_max = min(all_errors), max(all_errors)
margin = 0.05 * (y_max - y_min)
y_lim = (y_min - margin, y_max + margin)


fig, axes = plt.subplots(
    2, 2,
    figsize=(7.2, 6.6),
    sharex=True,
    sharey=True,
)

axes = axes.flatten()

x_formatter = ScalarFormatter(useMathText=True)
x_formatter.set_powerlimits((-3, -3))  

for ax, (name, (err_joint, err_sep)) in zip(
    axes, results.items()
):
    ax.plot(
        TOL_LIST,
        err_joint,
        marker="o",
        color=COLOR_PROPOSED,
        label="Structured", #  (last-dim preserved)
        markerfacecolor="white",
        markeredgewidth=1.5,
    )
    ax.plot(
        TOL_LIST,
        err_sep,
        marker="s",
        color=COLOR_BASELINE,
        label="Flattened", #  (last-dim merged)
        markerfacecolor="white",
        markeredgewidth=1.5,
    )

    ax.xaxis.set_major_locator(FixedLocator(TOL_LIST))
    ax.xaxis.set_major_formatter(x_formatter)

    ax.ticklabel_format(
        axis="x",
        style="sci",
        scilimits=(-3, -3),
        useMathText=True,
    )

    ax.set_xlim(
        TOL_LIST.min() - 0.2e-3,
        TOL_LIST.max() + 0.2e-3
    )

    ax.set_ylim(*y_lim)
    ax.set_title(name)

    ax.grid(True, axis="y")
    ax.grid(False, axis="x")

    ax.yaxis.set_major_formatter(
        FormatStrFormatter("%.3f")
    )


for ax in axes[2:]:
    ax.set_xlabel(r"$\mathrm{tol}$")

for ax in axes[::2]:
    ax.set_ylabel("Relative reconstruction error")


handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 1),
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig(
    OUT_DIR / "tol_error_all_datasets.pdf",
    bbox_inches="tight",
)
plt.savefig(
    OUT_DIR / "tol_error_all_datasets.jpg",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print("Saved tol–error figure.")
