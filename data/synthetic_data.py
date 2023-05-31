import numpy as np
from utils.utils import multivariate_normal_density

def create_gaussian_data(
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


def cond_prob_gaussian_one(
        x: np.array, 
        mu_1: np.array,
        mu_2: np.array,
        cov_1_inv: np.array,
        cov_1_det: float, 
        cov_2_inv: np.array,
        cov_2_det: float
    ) -> np.array:
    class_1_likelihood = multivariate_normal_density(
        x, mu_1, cov_1_inv, cov_1_det
    )
    class_2_likelihood = multivariate_normal_density(
        x, mu_2, cov_2_inv, cov_2_det
    )
    prob = (class_1_likelihood / (class_1_likelihood + class_2_likelihood))
    return prob

# The following functions approximate the conditional class probabilites of 
# the make_moons and make_circless function from scikit-learn
def unnormalized_gaussian_density(
        x: np.array,
        noise: float
    ) -> np.array:
    return np.exp(- 1 / (2 * (noise ** 2)) * np.linalg.norm(x, axis=1))


def unnormalized_inner_moon_density(
        x: np.array, 
        noise: float
    ) -> np.array:
    N = 1000
    outer_circ_x = np.cos(np.linspace(0, np.pi, N))
    outer_circ_y = np.sin(np.linspace(0, np.pi, N))

    moon_one = np.vstack([outer_circ_x, outer_circ_y]).T

    likelihoods = (1 / N) * unnormalized_gaussian_density(moon_one - x, noise)
    return np.sum(likelihoods)


def unnormalized_outer_moon_density(
        x: np.array,
        noise: float
    ) -> np.array:
    N = 1000
    outer_circ_x = 1 - np.cos(np.linspace(0, np.pi, N))
    outer_circ_y = 1 - np.sin(np.linspace(0, np.pi, N)) - .5

    moon_two = np.vstack([outer_circ_x, outer_circ_y]).T

    likelihoods = (1 / N) * unnormalized_gaussian_density(moon_two - x, noise)
    return np.sum(likelihoods)


def cond_prob_moon_one(
        x: np.array,
        noise: float
    ) -> np.array:
    n = len(x)
    cond_probs = np.zeros(n)
    for i in range(n):
        odds_1 = unnormalized_outer_moon_density(x[i, :], noise)
        odds_0 = unnormalized_inner_moon_density(x[i, :], noise)
        cond_probs[i] = odds_1 / (odds_0 + odds_1)

    return cond_probs 


def unnormalized_circle_inner_density(
        x: np.array,
        noise: float, 
        factor: float = 0.8
    ) -> np.array:
    N = 1000
    outer_circ_x = factor * np.cos(np.linspace(0, 2 * np.pi, N))
    outer_circ_y = factor * np.sin(np.linspace(0, 2 * np.pi, N))

    circle_inner = np.vstack([outer_circ_x, outer_circ_y]).T

    likelihoods = (1 / N) * unnormalized_gaussian_density(circle_inner - x, noise)
    return np.sum(likelihoods)


def unnormalized_circle_outer_density(
        x: np.array,
        noise: float
    ) -> np.array:
    N = 1000
    outer_circ_x = np.cos(np.linspace(0, 2 * np.pi, N))
    outer_circ_y = np.sin(np.linspace(0, 2 * np.pi, N))

    circle_inner = np.vstack([outer_circ_x, outer_circ_y]).T

    likelihoods = (1 / N) * unnormalized_gaussian_density(circle_inner- x, noise)
    return np.sum(likelihoods)


def cond_prob_circle_one(
        x: np.array,
        noise: float
    ) -> np.array:
    n = len(x)
    cond_probs = np.zeros(n)
    for i in range(n):
        odds_1 = unnormalized_circle_inner_density(x[i, :], noise)
        odds_0 = unnormalized_circle_outer_density(x[i, :], noise)
        cond_probs[i] = odds_1 / (odds_0 + odds_1)

    return cond_probs 


cond_prob_funcs = {
    'circles': cond_prob_circle_one, 
    'moons': cond_prob_moon_one, 
    'gaussians': cond_prob_gaussian_one
}
