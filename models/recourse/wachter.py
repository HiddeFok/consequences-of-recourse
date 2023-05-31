import scipy
import numpy as np

from tqdm import tqdm

from models.recourse.base import RecourseMethodBase

class Wachter(RecourseMethodBase):
    
    def __init__(self, model):
        super().__init__(model)

        self._eps = 0.05
        self._lambda_min = 1e-10
        self._lambda_max = 1e4
        self._lambda_steps = 30
        self._lambdas = np.logspace(
            np.log10(self._lambda_min), 
            np.log10(self._lambda_max)
        )

        self._p_target = 0.51

    @staticmethod
    def _cost(
        x_prime: np.array, 
        x: np.array, 
        y_prime: np.array, 
        lambda_value: float, 
        model
    ) -> float:
        distance = np.linalg.norm(x_prime - x)
        loss = (model.predict_proba(x_prime[None, :])[0, 1] - y_prime) ** 2
        return lambda_value * loss + distance

    def _provide_recourse_single_instance(
            self, 
            x: np.array
        ) -> np.array:
        candidates = []
        Y_primes = []
        for lambda_k in self._lambdas:
            args = (x, self._p_target, lambda_k, self._model)
            minimizer = scipy.optimize.minimize(
                self._cost, 
                x, 
                args=args
            )
            x_prime_hat = minimizer.x
            y_prime = self._model.predict_proba(x_prime_hat[None, :])
            Y_primes.append(y_prime[0, 1])
            candidates.append(x_prime_hat)

        Y_primes = np.array(Y_primes)
        candidates = np.array(candidates)

        # check if any counterfactual candidates meet the tolerance condition
        eps_condition = np.abs(Y_primes - self._p_target) <= self._eps
        
        if any(eps_condition):
            return candidates[eps_condition][-1, :]
        else:
            return np.zeros(x.shape)

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
                    print(f'Generated {i} CFs so far')
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

