"""
Custom implementation of the Growing Spheres counterfactual method. This method was derived and adapted
from the implementation in the CARLA recourse library. 
"""

import numpy as np

from tqdm import tqdm

from models.recourse.base import RecourseMethodBase

class GrowingSpheres(RecourseMethodBase):
    
    def __init__(self, model):
        super().__init__(model)

        self._n_search_samples=1000
        self._p_norm=2
        self._step=0.2
        self._max_iter=1000

    def _provide_recourse_single_instance(
            self,
            x: np.array
        ) -> np.array:
        return self._growing_spheres_search(x)

    def _provide_recourse(
            self,
            x: np.array, 
            y_hat: np.array, 
            pbar: bool = True
        ) -> np.array:
        x_recourse = np.zeros(x.shape)
        if pbar:
            pbar = tqdm(range(x_recourse.shape[0]), position=2, leave=False)
            pbar.set_description("CF search")
            for i in pbar:
                if y_hat[i] == 0:
                    cf = self._provide_recourse_single_instance(x[i, :])
                    x_recourse[i, :] = cf
                else:
                    x_recourse[i, :] = x[i, :]
        else: 
            for i in range(x_recourse.shape[0]):
                if i%50 == 0:
                    print(f"Generated {i} CFs so far")
                if y_hat[i] == 0:
                    cf = self._provide_recourse_single_instance(x[i, :])
                    x_recourse[i, :] = cf
                else:
                    x_recourse[i, :] = x[i, :]


        return x_recourse


    def provide_recourse(
            self, 
            x: np.array, 
            y_hat: np.array, 
            pbar: bool = True
        ) -> np.array:
        return self._provide_recourse(x, y_hat, pbar)


    def _hyper_sphere_coordinates(
            self,
            x: np.array,
            low: float,
            high: float
        ) -> np.array:
        """
        :param x: numpy input point array
       """

        delta_instance = np.random.randn(
            self._n_search_samples, 
            x.shape[1]
            )
        dist = np.random.rand(
            self._n_search_samples
        ) 
        # length range [l, h)
        dist = dist  * (high - low) + low  
        norm_p = np.linalg.norm(delta_instance, ord=self._p_norm, axis=1)
        d_norm = np.divide(dist, norm_p).reshape(-1, 1)  # rescale/normalize factor
        delta_instance = np.multiply(delta_instance, d_norm)
        candidate = x + delta_instance

        return candidate, dist

    def _growing_spheres_search(
            self,
            x: np.array
        ):
        
        instance_replicated = np.repeat(
            x[None, :], 
            self._n_search_samples, 
            axis=0
        ) 

        # init step size for growing the sphere
        low = 0
        high = low + self._step

        # counter
        count = 0
        counter_step = 1

        # get predicted label of instance
        instance_label = np.argmax(self._model.predict_proba(x[None, :]))

        counterfactuals_found = False
        candidate_counterfactual_star = x

        while (not counterfactuals_found) and (count < self._max_iter):
            count = count + counter_step

            candidate_counterfactuals, _ = self._hyper_sphere_coordinates(
                instance_replicated, 
                high, 
                low
            )

            if self._p_norm == 1:
                distances = np.abs(
                    (candidate_counterfactuals - instance_replicated)
                ).sum(axis=1)
            elif self._p_norm == 2:
                distances = np.square(
                    (candidate_counterfactuals - instance_replicated)
                ).sum(axis=1)
            else:
                raise ValueError("Distance not defined yet")

            # counterfactual labels
            y_candidate_logits = self._model.predict_proba(
                candidate_counterfactuals
            )
            y_candidate = np.argmax(y_candidate_logits, axis=1)
            indices = np.where(y_candidate != instance_label)
            candidate_counterfactuals = candidate_counterfactuals[indices]
            candidate_dist = distances[indices]

            if len(candidate_dist) > 0:  # certain candidates generated
                min_index = np.argmin(candidate_dist)
                candidate_counterfactual_star = candidate_counterfactuals[min_index]
                counterfactuals_found = True

            # no candidate found & push search range outside
            low = high
            high = low + self._step

        return candidate_counterfactual_star