import os
from typing import Callable, Union

import numpy as np
from scipy import linalg

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors


cmap = colors.LinearSegmentedColormap(
    "red_blue_classes",
    {
        "red": [(0, 1, 1), (1, 0.7, 0.7)],
        "green": [(0, 0.7, 0.7), (1, 0.7, 0.7)],
        "blue": [(0, 0.7, 0.7), (1, 1, 1)],
    },
)

def plot_ellipse(
        axs: mpl.pyplot.Axes,
        mean: np.array,
        cov: np.array,
        color: colors.LinearSegmentedColormap
    ) -> None:
    v, w = linalg.eigh(cov)
    u = w[0] / linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi

    ell = mpl.patches.Ellipse(
        mean,
        2 * v[0] ** 0.5,
        2 * v[1] ** 0.5,
        angle=180 + angle,
        facecolor=color,
        edgecolor="black",
        linewidth=1
        )
    ell.set_clip_box(axs.bbox)
    ell.set_alpha(0.2)
    axs.add_artist(ell)
    axs.set_xticks(())
    axs.set_yticks(())


# Storing figures
def check_save_fig(
        fig: plt.Figure,
        checkpoint_dir: Union[str, bytes, os.PathLike], 
        fname: str
    ) -> None:
    fig.savefig(os.path.join(checkpoint_dir, f"figures/{fname}.pdf"))
    fig.savefig(
        os.path.join(checkpoint_dir, f"figures/{fname}.png"), 
        dpi=600, 
        transparent=False, 
        bbox_inches="tight"
        )


def plot_with_probs(
            predict_proba: Callable,
            X: np.array,
            y: np.array,
            axs: plt.Axes,
            title: str
    ) -> None:
        x_min, x_max = 1.2 * X[:, 0].min(), 1.2 * X[:, 0].max()
        y_min, y_max = 1.2 * X[:, 1].min(), 1.2 * X[:, 1].max()

        x_1 = X[y == 1, :]
        x_2 = X[y == 0, :]

        axs.scatter(
            x_1[:, 0], x_1[:, 1],
            label="Class 1", marker=".", color="blue"
        )
        axs.scatter(
            x_2[:, 0], x_2[:, 1],
            label="Class 2", marker=".", color="red"
        )
        # Class areas
        nx, ny = 20, 20
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, nx),
            np.linspace(y_min, y_max, ny)
        )
        Z = predict_proba(np.c_[xx.ravel(), yy.ravel()])
        xyz = np.c_[xx.ravel(), yy.ravel(), Z]
        xyz = xyz[np.lexsort((xyz[:, 1], xyz[:, 0]))]

        if len(Z.shape) == 2:
            Z = Z[:, 1].reshape(xx.shape)
        else:
            Z = Z.reshape(xx.shape)

        axs.pcolormesh(
            xx, yy, Z, cmap=cmap, norm=colors.Normalize(0.0, 1.0), zorder=0, shading="auto"
        )
        axs.contour(xx, yy, Z, [0.5], linewidths=1.0, colors="white")
        axs.set_xlim(x_min, x_max)
        axs.set_ylim(y_min, y_max)
        axs.set_title(title)

        axs.legend()
        return xyz


def plot_predictions_with_probs(
            predict_proba: Callable,
            X_train: np.array, 
            X: np.array,
            y: np.array,
            y_hat: np.array,
            axs: plt.Axes,
            title: str
    ) -> None:
        x_min, x_max = 1.2 * X_train[:, 0].min(), 1.2 * X_train[:, 0].max()
        y_min, y_max = 1.2 * X_train[:, 1].min(), 1.2 * X_train[:, 1].max()

        x_1 = X[y == 1, :]
        x_2 = X[y == 0, :]

        y_1 = y_hat[y == 1]
        y_2 = y_hat[y == 0]

        # Class 1
        x_1_correct = x_1[y_1 == 1, :]
        x_1_false = x_1[y_1 == 0, :]

        # Class 2
        x_2_correct = x_2[y_2 == 0, :]
        x_2_false = x_2[y_2 == 1, :]

        axs.scatter(
            x_1_correct[:, 0], x_1_correct[:, 1],
            label="Class 1", marker=".", color="blue"
        )
        axs.scatter(
            x_2_correct[:, 0], x_2_correct[:, 1],
            label="Class 2", marker=".", color="red"
        )

        axs.scatter(
            x_1_false[:, 0], x_1_false[:, 1],
            label="Class 1", marker="x", color="#000099"
        )
        axs.scatter(
            x_2_false[:, 0], x_2_false[:, 1],
            label="Class 2", marker="x", color="#990000"
        )
        # Class areas
        nx, ny = 20, 20
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, nx),
            np.linspace(y_min, y_max, ny)
        )
        Z = predict_proba(np.c_[xx.ravel(), yy.ravel()])
        if len(Z.shape) == 2:
            z = Z[:, 1].copy()
            Z = Z[:, 1].reshape(xx.shape)
        else:
            z = Z.copy()
            Z = Z.reshape(xx.shape)

        xyz = np.c_[xx.ravel(), yy.ravel(), z]
        xyz = xyz[np.lexsort((xyz[:, 1], xyz[:, 0]))]

        db_coords = xyz[
            np.isclose(xyz[:, 2], .5, atol=1e-3), 
            :][:, 0:2]

        axs.pcolormesh(
            xx, yy, Z, cmap=cmap, norm=colors.Normalize(0.0, 1.0), zorder=0, shading="auto"
        )
        axs.contour(xx, yy, Z, [0.5], linewidths=1.0, colors="white")
        axs.set_xlim(x_min, x_max)
        axs.set_ylim(y_min, y_max)
        axs.set_xticks(())
        axs.set_yticks(())
        axs.set_title(title)

        axs.legend()
        return xyz, db_coords
