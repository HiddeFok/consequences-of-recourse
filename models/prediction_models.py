"""
Implementations of LDA and QDA with some custom plot and parameter functions
"""

from abc import ABC, abstractmethod
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix

from matplotlib import colors


from utils.utils import local_norm_sq, multivariate_normal_density, projection
from utils.plot_utils import cmap, plot_ellipse


class Classifier(ABC):
    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def predict_proba(self, x: np.array) -> np.array:
        pass

    @abstractmethod
    def plot_ellipses(self, axs: plt.Axes) -> None:
        pass

    @abstractmethod
    def provide_recourse(self, x: np.array, y_hat: np.array) -> np.array:
        pass

    def _plot(
            self,
            predict_proba: Callable,
            X: np.array,
            y: np.array,
            y_hat: np.array,
            axs: plt.Axes,
            title: str
    ) -> None:
        x_max = 1.2 * max(X[:, 0].max(), X[:, 1].max())

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
            np.linspace(-x_max, x_max, nx),
            np.linspace(-x_max, x_max, ny)
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
        axs.set_xlim(-x_max, x_max)
        axs.set_ylim(-x_max, x_max)
        axs.set_title(title)

        # Class means
        axs.plot(
            self.mu_1[0], self.mu_1[1],
            "*",
            color="yellow",
            markersize=15,
            markeredgecolor="grey"
        )
        axs.plot(
            self.mu_2[0], self.mu_2[1],
            "*",
            color="yellow",
            markersize=15,
            markeredgecolor="grey"
        )

        self.plot_ellipses(axs)
        axs.legend()

        return xyz

    def plot(
            self,
            X: np.array,
            y: np.array,
            y_hat: np.array,
            axs: plt.Axes,
            title: str
    ):
        xyz = self._plot(self.predict_proba, X, y, y_hat, axs, title)
        return xyz


class LinearDiscriminantAnalysis(Classifier):
    def __init__(
            self,
            mu_1: np.array,
            mu_2: np.array,
            cov: np.array,
            frac: float = 0.5
    ):
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.cov = cov
        self.cov_inv = np.linalg.inv(cov)
        self.frac = frac

        self.beta, self.c = self.get_params()

    def get_params(self):
        beta = self.cov_inv @ (self.mu_1 - self.mu_2)

        frac_term = np.log(self.frac / (1 - self.frac))
        c = 0.5 * np.dot(beta, (self.mu_1 + self.mu_2)) - frac_term
        return beta, c

    @staticmethod
    def _predict(x: np.array, beta: np.array, c: float) -> np.array:
        log_likelihood_ratio = np.dot(x, beta)

        eps = np.finfo(np.float32).eps
        lower_bound = log_likelihood_ratio - c - eps >= 0
        upper_bound = log_likelihood_ratio - c + eps >= 0
        return np.logical_or(lower_bound, upper_bound).astype(int)

    def predict(self, x: np.array) -> np.array:
        return self._predict(x, self.beta, self.c)

    @staticmethod
    def _provide_recourse(x: np.array, y_hat: np.array, beta: np.array, c: float) -> np.array:
        x_after_recourse = x.copy()

        x_after_recourse[y_hat == 0] = projection(x_after_recourse[y_hat == 0], beta, c)
        return x_after_recourse

    def provide_recourse(self, x: np.array, y_hat: np.array) -> np.array:
        return self._provide_recourse(x, y_hat, self.beta, self.c)

    def predict_proba(self, x: np.array) -> np.array:
        p = 1 / (1 + np.exp(- (np.dot(self.beta, x.T) - self.c)))
        return p

    def plot_ellipses(self, axs: plt.Axes) -> None:
        plot_ellipse(axs, self.mu_1, self.cov, "blue")
        plot_ellipse(axs, self.mu_2, self.cov, "red")


class QuadraticDiscriminantAnalysis(Classifier):
    def __init__(
            self,
            mu_1: np.array,
            mu_2: np.array,
            cov_1: np.array,
            cov_2: np.array,
            frac: float = 0.5
    ):
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.cov_1 = cov_1
        self.cov_2 = cov_2
        self.cov_1_inv = np.linalg.inv(cov_1)
        self.cov_2_inv = np.linalg.inv(cov_2)
        self.cov_1_det = np.linalg.det(cov_1)
        self.cov_2_det = np.linalg.det(cov_2)

        self.frac = frac

    def predict(self, x: np.array) -> np.array:
        log_det_1 = np.log(self.cov_1_det)
        log_det_2 = np.log(self.cov_2_det)
        log_likelihood_1 = local_norm_sq(x, self.mu_1, self.cov_1_inv) + log_det_1
        log_likelihood_2 = local_norm_sq(x, self.mu_2, self.cov_2_inv) + log_det_2

        eps = np.finfo(np.float32).eps

        log_likelihood_ratio = log_likelihood_1 - log_likelihood_2
        lower_bound = log_likelihood_ratio - eps <= 0
        upper_bound = log_likelihood_ratio + eps <= 0
        return np.logical_or(lower_bound, upper_bound).astype(int)

    def _generate_recourse_candidates(self) -> np.array:
        nx, ny = 200, 200
        xx, yy = np.meshgrid(np.linspace(-10, 10, nx), np.linspace(-10, 10, ny))
        Z = self.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        candidates = np.c_[
            xx[(0.5 <= Z) & (Z <= 0.51)].ravel(),
            yy[(0.5 <= Z) & (Z <= 0.51)].ravel()
        ]
        return candidates

    def _provide_recourse(self, x: np.array, y_hat: np.array) -> np.array:
        x_1_predicted = x[y_hat == 1, :]
        x_2_predicted = x[y_hat == 0, :]

        candidates = self._generate_recourse_candidates()
        dist_matrix = distance_matrix(x_2_predicted, candidates)
        minimizers_idx = np.argmin(dist_matrix, axis=1)
        x_2_recourse = candidates[minimizers_idx, :]
        x_after_recourse = np.vstack([x_1_predicted, x_2_recourse])
        return x_after_recourse

    def provide_recourse(self, x: np.array, y_hat: np.array) -> np.array:
        return self._provide_recourse(x, y_hat)

    def predict_proba(self, x: np.array) -> np.array:
        class_1_likelihood = multivariate_normal_density(
            x, self.mu_1, self.cov_1_inv, self.cov_1_det
        )
        class_2_likelihood = multivariate_normal_density(
            x, self.mu_2, self.cov_2_inv, self.cov_2_det
        )
        prob = (class_1_likelihood / (class_1_likelihood + class_2_likelihood))
        return prob.squeeze()

    def plot_ellipses(self, axs: plt.Axes) -> None:
        plot_ellipse(axs, self.mu_1, self.cov_1, "blue")
        plot_ellipse(axs, self.mu_2, self.cov_2, "red")


