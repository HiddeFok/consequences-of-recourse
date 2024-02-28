import os 
import random

from typing import Union

from datetime import datetime
from typing import Callable, Union

import numpy as np


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def setup_checkpoint_dir(prefix: str) -> Union[str, bytes, os.PathLike]:
    # Setup checkpoint dir to log settings and store results
    current_date = datetime.now()

    checkpoint_dir = "checkpoints/{}_{:02d}_{:02d}_{:02d}__{:02d}_{:02d}/".format(
        prefix,
        current_date.year, current_date.month, current_date.day,
        current_date.hour, current_date.minute
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(checkpoint_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(checkpoint_dir, "data"), exist_ok=True)

    return checkpoint_dir


def multivariate_normal_density(
        x: np.array,
        mean: np.array,
        cov_inv: np.array,
        cov_det: float
    ) -> np.array:
    C = 1 / (2 * np.pi * cov_det)
    exp = np.exp(-.5 * local_norm_sq(x, mean, cov_inv))
    return C * exp


def create_synthetic_data(
        mean_1: np.array,
        mean_2: np.array,
        cov_1: np.array,
        cov_2: np.array,
        N: int,
        frac: float = 0.5
) -> np.array:

    features_class_1 = np.random.multivariate_normal(
        mean=mean_1,
        cov=cov_1,
        size=int(N * frac)
    )
    features_class_2 = np.random.multivariate_normal(
        mean=mean_2,
        cov=cov_2,
        size=int(N * (1 - frac))
    )

    features = np.vstack((features_class_1, features_class_2))
    labels = np.zeros((N, 1))
    labels[:(int(N * frac))] = 1
    data = np.hstack((features, labels))

    return data


def sample_bernoulli(n: int, p: Union[float, np.array]) -> np.array:
    if hasattr(p, "__len__"):
        return (np.random.uniform(size=n) < p).astype(np.uint8)
    else:
        p = np.repeat(p, n)
        return (np.random.uniform(size=n) < p).astype(np.uint8)


def projection(x: np.array, beta: np.array, c: np.array) -> np.array:
    s = np.array([[beta[1], -beta[0]]])
    affine_correction = c / (2 * beta)
    scalar = np.dot(x - affine_correction, s.T) / np.dot(s, s.T)
    return scalar * s + affine_correction


def logistic(x: np.array, beta: np.array, c: np.array) -> np.array:
    p = 1 / (1 + np.exp(- (np.dot(beta, x.T) - c)))
    return p


def empirical_risk(y: np.array, y_hat: np.array) -> np.array:
    risk = (y != y_hat).astype(int).mean()
    return risk


def sample_class(p: np.array) -> np.array:
    n = p.shape[0]
    u = np.random.uniform(size=n)
    return (u < p).astype(int)


def resample_classes(x: np.array, prob_func: Callable) -> np.array:
    p_y_given_x = prob_func(x)
    if len(p_y_given_x.shape) == 2:
        y_new = sample_class(p_y_given_x[:, 1])
    else:
        y_new = sample_class(p_y_given_x)

    return y_new


def local_norm_sq(x: np.array, y: np.array, A: np.array) -> np.array:
    if len(x.shape) == 2:
        return np.sum((x - y) * np.dot(A, (x - y).T).T, axis=1)
    else:
        return np.sum(x - y) * np.dot(A, (x - y).T).T


def tp_fn_fp_tn(X, y, y_hat):
    x_1 = X[y == 1, :]
    x_0 = X[y == 0, :]

    y_1 = y_hat[y == 1]
    y_0 = y_hat[y == 0]

    x_11 = x_1[y_1 == 1, :]
    x_10 = x_1[y_1 == 0, :]
    x_01 = x_0[y_0 == 1, :]
    x_00 = x_0[y_0 == 0, :]

    return x_11, x_10, x_01, x_00
