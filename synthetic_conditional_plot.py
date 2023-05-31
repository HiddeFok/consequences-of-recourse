import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from data.synthetic_data import cond_prob_funcs

from utils.experiment import save_x_y_data
from utils.plot_utils import plot_with_probs, check_save_fig
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

SEED = 124
TEST_SIZE = 1000
set_seed(SEED)

checkpoint_dir = f"checkpoints/cond_probs"
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(os.path.join(checkpoint_dir, "figures"), exist_ok=True)
os.makedirs(os.path.join(checkpoint_dir, "data"), exist_ok=True)

for data in ['moons', 'circles', 'gaussians']: 
    exp_args = EXPERIMENT_SETTINGS[data]['args'].values()
    exp_kwargs = EXPERIMENT_SETTINGS[data]['kwargs']

    X, y = load_synthetic_data(data, *exp_args, **exp_kwargs)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)

    if data in ['moons', 'circles']:
        cond_prob_func = lambda x: cond_prob_funcs[data](
            x, noise=EXPERIMENT_SETTINGS[data]['kwargs']['noise']
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
        cond_prob_func = lambda x: cond_prob_funcs[data](
            x, 
            mean_1, mean_2, 
            cov_1_inv, cov_1_det, 
            cov_2_inv, cov_2_det 
            )



    ## Plotting and saving
    print(f"Plotting and saving figures: {data}, {data}, {data}")
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    xyz = plot_with_probs(
        cond_prob_func, 
        X_test, y_test,
        ax, 
        title=""
    )
    fig.tight_layout()
    check_save_fig(fig, checkpoint_dir, f"predictions_{data}")
    fig.clear()

    data_dir = os.path.join(checkpoint_dir, f"data/{data}_cond_probs")
    save_x_y_data(data_dir, X_test, y_test, y_test)
    np.savetxt(os.path.join(data_dir, "xyz_cond_probs.dat"), 
            xyz, fmt='%.6f')
    np.savetxt(os.path.join(data_dir, "db_coords.dat"), 
            np.array([]), fmt='%.6f')