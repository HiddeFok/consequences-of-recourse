import numpy as np
from scipy.spatial import distance_matrix

from models.recourse.base import RecourseMethodBase
from utils.utils import sample_bernoulli


class BruteForceRecourse(RecourseMethodBase):

    def __init__(self, model):
        super().__init__(model)

    def _generate_recourse_candidates(self) -> np.array:
        nx, ny = 400, 400
        xx, yy = np.meshgrid(np.linspace(-4, 4, nx), np.linspace(-4, 4, ny))
        Z = self._model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        if len(Z.shape) == 2:
            Z = Z[:, 1].reshape(xx.shape)
        else:
            Z = Z.reshape(xx.shape)

        candidates = np.c_[
            xx[(0.5 <= Z) & (Z <= 1)].ravel(),
            yy[(0.5 <= Z) & (Z <= 1)].ravel()
        ]
        return candidates


    def _provide_recourse(
            self, 
            x: np.array, 
            y_hat: np.array, 
            prob: float or None = None,
            sigma: float or None = None,
            bpar: bool or None = None
        ) -> np.array:
        x_after_recourse = x.copy()
        x_2_predicted = x[y_hat == 0, :]

        candidates = self._generate_recourse_candidates()

        dist_matrix = distance_matrix(x_2_predicted, candidates)
        minimizers_idx = np.argmin(dist_matrix, axis=1)

        x_2_recourse = candidates[minimizers_idx, :]
        x_after_recourse[y_hat == 0, :] = x_2_recourse
        return x_after_recourse

    def provide_recourse(
            self, 
            x: np.array,
            y_hat: np.array,
            pbar: bool or None = None
        ) -> np.array:
        return self._provide_recourse(x, y_hat)
