import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import mean_squared_error
from fABBA import JABBA

def load_images(folder, shape=(300, 250)):
    images = []
    for filename in sorted(os.listdir(folder)):
        path = os.path.join(folder, filename)
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, shape)
        images.append(img.astype(np.float32))
    return np.array(images)


def jabba_image_compress(
    img,
    *,
    last_dim: bool,
    tol: float = 0.5,
    k: int = 200
):
    jabba = JABBA(
        tol=tol,
        k=k,
        init="kmeans",
        last_dim=last_dim,
        verbose=0
    )
    symbols = jabba.fit_transform(img)
    img_rec = jabba.inverse_transform(symbols)
    return np.clip(img_rec, 0, 255).astype(np.uint8)

def run_image_experiment(
    data,
    *,
    tol=0.5,
    k=200,
    save_fig="results/images/jabba_image_compare.pdf",
    save_csv="results/infoIMG_jabba_lastdim.csv"
):
    os.makedirs(os.path.dirname(save_fig), exist_ok=True)

    results = []

    n_show = min(24, len(data))
    n_cols = min(8, n_show)
    n_rows = 5

    fig, axs = plt.subplots(
        n_rows, n_cols,
        figsize=(20, 12),
        squeeze=False
    )

    row_titles = [
        "Original",
        "Flattened",
        "Structured",
        "Difference \n(Flattened)",
        "Difference \n(Structured)"
    ]

    y_positions = np.linspace(0.86, 0.14, n_rows)
    for y, title in zip(y_positions, row_titles):
        fig.text(
            0.015, y, title,
            fontsize=18,
            va="center",
            ha="left"
        )

    diff_im = None  

    for idx in tqdm(range(n_show)):
        col = idx % n_cols
        img = data[idx]

        # Original
        axs[0, col].imshow(img.astype(np.uint8))
        axs[0, col].axis("off")

        # Flattened
        rec_sep = jabba_image_compress(
            img, last_dim=False, tol=tol, k=k
        )
        axs[1, col].imshow(rec_sep)
        axs[1, col].axis("off")

        # Diff (grayscale absolute error, normalized)
        diff = np.abs(img - rec_sep)
        diff = diff.mean(axis=2)          # RGB → gray error
        diff = diff / (diff.max() + 1e-8) # normalize [0, 1]

        diff_im = axs[2, col].imshow(
            diff,
            cmap="gray",
            vmin=0,
            vmax=1
        )
        axs[2, col].axis("off")

        # Structured
        rec_joint = jabba_image_compress(
            img, last_dim=True, tol=tol, k=k
        )
        axs[3, col].imshow(rec_joint)
        axs[3, col].axis("off")

        # Diff (grayscale absolute error, normalized)
        diff = np.abs(img - rec_joint)
        diff = diff.mean(axis=2)          # RGB → gray error
        diff = diff / (diff.max() + 1e-8) # normalize [0, 1]

        diff_im = axs[4, col].imshow(
            diff,
            cmap="gray",
            vmin=0,
            vmax=1
        )
        axs[4, col].axis("off")

        results.append({
            "image_id": idx,
            "tol": tol,
            "k": k,
            "mse_last_dim_false": mean_squared_error(img, rec_sep),
            "mse_last_dim_true": mean_squared_error(img, rec_joint),
        })

    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  
    cbar = fig.colorbar(diff_im, cax=cax)
    cbar.set_label("Normalized Absolute Error", fontsize=14)

    plt.tight_layout(rect=[0.08, 0.02, 0.90, 0.98])

    plt.savefig(save_fig, dpi=300, bbox_inches="tight")
    plt.show()

    df = pd.DataFrame(results)
    df.to_csv(save_csv, index=False)
    print(f"Saved results to {save_csv}")

    return df

if __name__ == "__main__":
    data = load_images("image_samples")
    run_image_experiment(
        data,
        tol=0.5,
        k=200,
        save_fig="results/images/jabba_image_compare_tol0.5.pdf",
        save_csv="results/infoIMG_jabba_lastdim_tol0.5.csv"
    )
