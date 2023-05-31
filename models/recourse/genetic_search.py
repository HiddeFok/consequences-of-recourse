import numpy as np
from tqdm import tqdm 

from cogs.evolution import Evolution
from cogs.fitness import gower_fitness_function

from models.recourse.base import RecourseMethodBase

from utils.utils import resample_classes, empirical_risk


class GeneticSearch(RecourseMethodBase):

    def __init__(self, model):
        super().__init__(model)

        self._feature_interval = [(0, 1)] * self._model.n_features_in_ 

    def _provide_recourse_single_instance(self, x) -> np.array:
        cogs = Evolution(
        # hyper-parameters of the problem (required!)
        x=x,  
        fitness_function=gower_fitness_function,  
        fitness_function_kwargs={
            'blackbox':self._model,
            'desired_class': 1 
        },  
        feature_intervals=self._feature_interval,
        indices_categorical_features=[],
        plausibility_constraints=None, 
        # hyper-parameters of the evolution (all optional)
        evolution_type='classic', 
        population_size=1000,   
        n_generations=100,  
        selection_name='tournament_4', 
        init_temperature=0.8, 
        num_features_mutation_strength=0.25, 
        num_features_mutation_strength_decay=0.5, 
        num_features_mutation_strength_decay_generations=[50,75,90], 
        verbose=False  
        )
        cogs.run()
        return cogs.elite

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
                    print(f"generated {i} CFs so far")
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
            pbar: bool = True):
        return self._provide_recourse(x, y_hat, pbar)


def recourse_multiple_models(X_test, predictions, models, cond_proba, recourse_class):
    counterfactuals = {}
    predictions_after_recourse = {}
    risk_after_recourse = {}
    y_after_recourse = {}

    pbar = tqdm(models)
    for clf in pbar:
        pbar.set_description(
            f"Generating counterfactuals for {models[clf]['name']}"
        )

        recourse_method = recourse_class(models[clf]['model'])

        counterfactuals[clf] = recourse_method(
            X_test, 
            predictions[clf], 
            models[clf]["model"].predict_proba
        )
        predictions_after_recourse[clf] = models[clf]["model"].predict(
            counterfactuals[clf]
        )
        y_after_recourse[clf] = resample_classes(
            counterfactuals[clf],
            cond_proba
        )
        risk_after_recourse[clf] = empirical_risk(
            predictions_after_recourse[clf],
            y_after_recourse[clf]
        )

    return counterfactuals, predictions_after_recourse, risk_after_recourse, y_after_recourse