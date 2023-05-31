import os
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser

from sklearn.model_selection import train_test_split

from data.synthetic_data import cond_prob_funcs

from models.models_experiments import models
from models.models_experiments import recourse_models_synthetic as recourse_models

from utils.plot_utils import plot_predictions_with_probs, check_save_fig
from utils.experiment import do_single_synthetic_experiment, save_single_experiment_data
from utils.preproc import load_synthetic_data
from utils.utils import set_seed

EXPERIMENT_SETTINGS = {
    'circles': {
        'args': {
            'N': 5000, 
        },
        'kwargs':{
            'factor': 0.6, 
            'noise': 0.2, 
        }, 
   },
    'moons': {
        'args': {
            'N': 5000, 
        }, 
        'kwargs': {
            'noise': 0.2, 
        }, 
   },
    'gaussians': {
        'args': {
            "mean_1": np.array([1.0, 0.]),
            "mean_2": np.array([-1.0, 0.0]),
            "cov_1": np.array([[1., -.3], [-.3, 1.]]),
            "cov_2": np.array([[1., .8], [.8, 1.]]), 
            'N': 5000,
        }, 
        'kwargs': {

        }
    }
}

if __name__ == "__main__":
    SEED = 124
    TEST_SIZE = 1000
    set_seed(SEED)

    parser = ArgumentParser()
    parser.add_argument('--data_set', type=str,
                        choices=['moons', 'circles', 'gaussians'])
    parser.add_argument('--classifier', type=str, 
                        choices=list(models.keys()))
    parser.add_argument('--recourse', type=str,
                        choices=list(recourse_models.keys()))

    args = parser.parse_args()
    checkpoint_dir = f"checkpoints/{args.data_set}_cluster"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(checkpoint_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(checkpoint_dir, "data"), exist_ok=True)

    exp_args = EXPERIMENT_SETTINGS[args.data_set]['args'].values()
    exp_kwargs = EXPERIMENT_SETTINGS[args.data_set]['kwargs']

    X, y = load_synthetic_data(args.data_set, *exp_args, **exp_kwargs)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)

    if args.data_set in ['moons', 'circles']:
        cond_prob_func = lambda x: cond_prob_funcs[args.data_set](
            x, noise=EXPERIMENT_SETTINGS[args.data_set]['kwargs']['noise']
            )
    else:
        mean_1 = EXPERIMENT_SETTINGS['gaussians']['args']['mean_1']
        mean_2 = EXPERIMENT_SETTINGS['gaussians']['args']['mean_2']
        cov_1 = EXPERIMENT_SETTINGS['gaussians']['args']['cov_1']
        cov_2 = EXPERIMENT_SETTINGS['gaussians']['args']['cov_2'] 
        cov_1_inv = np.linalg.inv(cov_1)
        cov_2_inv = np.linalg.inv(cov_2)
        cov_1_det = np.linalg.det(cov_1)
        cov_2_det = np.linalg.det(cov_2)
        cond_prob_func = lambda x: cond_prob_funcs[args.data_set](
            x, 
            mean_1, mean_2, 
            cov_1_inv, cov_1_det, 
            cov_2_inv, cov_2_det 
            )

    classifier = models[args.classifier]
    recourse = recourse_models[args.recourse]

    print(f"Starting Experiment: {args.data_set}, {args.classifier}, {args.recourse}")
    result_dict = do_single_synthetic_experiment(
        X_train, X_test,
        y_train, y_test,
        cond_prob_func, 
        classifier, 
        recourse
    )

    ## Plotting and saving
    print(f"Plotting and saving figures: {args.data_set}, {args.classifier}, {args.recourse}")
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    xyz, db_coords = plot_predictions_with_probs(
        classifier['model'].predict_proba, 
        X_train,
        X_test, y_test,
        result_dict['predictions'], 
        ax, 
        title=""
    )
    fig.tight_layout()
    check_save_fig(fig, checkpoint_dir, f"predictions_{args.classifier}")
    fig.clear()

    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    xyz, db_coords = plot_predictions_with_probs(
        classifier['model'].predict_proba, 
        X_train,
        result_dict['counterfactuals'], 
        result_dict['y_after_recourse'],
        result_dict['predictions_after_recourse'], 
        ax, 
        title=""
    )
    fig.tight_layout()
    check_save_fig(fig, checkpoint_dir, f"predictions_{args.classifier}_{args.recourse}")
    fig.clear()

    print(f"Saving final data: {args.data_set}, {args.classifier}, {args.recourse}")
    save_single_experiment_data(
        checkpoint_dir,
        result_dict, 
        args.classifier,
        args.recourse,
        xyz, 
        db_coords,
        two_dim=True
    )
