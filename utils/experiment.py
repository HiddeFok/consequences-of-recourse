import os
import numpy as np
from typing import Dict, Callable

from joblib import load, dump

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV 
from sklearn.model_selection import GridSearchCV

from utils.preproc import load_data
from utils.utils import empirical_risk, resample_classes, tp_fn_fp_tn


def do_single_real_experiment(
        dataset_name: str,
        fracs: Dict[str, float],
        classifier,
        recourse_method
    ) -> Dict:
    data = load_data(dataset_name, fracs)

    Xs, ys = data['Xs'], data['ys']

    ## Conditional Probability Estimation ## 
    cond_prob_dir = f'models/fitted/{dataset_name}_cond_prob.joblib'
    if os.path.exists(cond_prob_dir):
        print("--- Loading pre-fitted cond proba model ---")
        cond_prob_estimator = load(cond_prob_dir)
    else:
        print("--- No pre-fitted cond proba model found, fitting now ---")
        gbc = GradientBoostingClassifier()

        gbc_parameters = {
            'learning_rate': [0.05,  0.15], 
            'n_estimators': [10, 20, 60], 
            'subsample': [1, 0.8, 0.9],
            'max_depth': [1, 2, 3]

        }
        cond_prob_estimator = GridSearchCV(
            estimator=gbc, 
            param_grid=gbc_parameters, 
            verbose=2
        ) 
        cond_prob_estimator.fit(Xs["cond_train"], ys["cond_train"])

        cond_prob_estimator = CalibratedClassifierCV(cond_prob_estimator, cv="prefit")
        cond_prob_estimator.fit(Xs["cond_calib"], ys["cond_calib"])
        dump(cond_prob_estimator, cond_prob_dir)


    ## Classifcation & Counterfactual generation ##

    ## Classification ##
    classifier['model'].fit(Xs['train'], ys['train'])
    predictions = classifier['model'].predict(Xs['test'])
    risk = empirical_risk(
        ys['test'], 
        predictions
    )

    ## Provide Recourse ##
    recourse_method = recourse_method(classifier["model"])

    counterfactuals = recourse_method.provide_recourse(
        Xs['test'], 
        predictions, 
        pbar=False
    )
    predictions_after_recourse = classifier['model'].predict(
        counterfactuals
    )
    y_after_recourse = ys['test'].copy()
    y_after_recourse[predictions == 0] = resample_classes(
        counterfactuals[predictions == 0],
        cond_prob_estimator.predict_proba
    )
    risk_after_recourse = empirical_risk(
        predictions_after_recourse,
        y_after_recourse
    )

    return_dict = {
        'X': Xs['test'], 
        'y': ys['test'],
        'risks': risk, 
        'predictions': predictions, 
        'counterfactuals': counterfactuals, 
        'predictions_after_recourse': predictions_after_recourse, 
        'y_after_recourse': y_after_recourse, 
        'risk_after_recourse': risk_after_recourse 
    }
    return return_dict


def do_single_synthetic_experiment(
    X_train: np.array, X_test: np.array, 
    y_train: np.array, y_test: np.array,
    cond_prob_func: Callable, 
    classifier, 
    recourse_method
) -> Dict:

    ## Classifcation ##
    classifier['model'].fit(X_train, y_train)
    predictions = classifier['model'].predict(X_test)
    risks = empirical_risk(
        y_test, 
        predictions
    )

    ## Provide Recourse ##
    recourse_method = recourse_method(classifier['model'])
    counterfactuals = recourse_method.provide_recourse(
        X_test, 
        predictions, 
        pbar=False
    )
    predictions_after_recourse = classifier['model'].predict(
        counterfactuals
    )
    y_after_recourse = y_test.copy()
    y_after_recourse[predictions == 0] = resample_classes(
        counterfactuals[predictions == 0, :],
        cond_prob_func
    )
    risk_after_recourse = empirical_risk(
        predictions_after_recourse,
        y_after_recourse
    )

    return_dict = {
        'X': X_test, 
        'y': y_test,
        'risks': risks, 
        'predictions': predictions, 
        'counterfactuals': counterfactuals, 
        'predictions_after_recourse': predictions_after_recourse, 
        'y_after_recourse': y_after_recourse, 
        'risk_after_recourse': risk_after_recourse 
    }
    return return_dict


def save_single_experiment_data(
        checkpoint_dir: str,
        result_dict: Dict, 
        classifier: str, 
        recourse: str,
        xyz_cond_probs: np.array, 
        db_coords: np.array,
        two_dim: bool = False
    ) -> None:
    X = result_dict['X']
    y = result_dict['y']

    predictions = result_dict['predictions']
    risk_before = result_dict['risks']

    risk_after= result_dict['risk_after_recourse']
    counterfactuals = result_dict['counterfactuals']
    predictions_after_recourse = result_dict['predictions_after_recourse']
    y_after_recourse = result_dict['y_after_recourse']
 
    data_dir = os.path.join(checkpoint_dir, 'data')
    np.savetxt(os.path.join(data_dir, "X.dat"), X, fmt='%.6f')
    np.savetxt(os.path.join(data_dir, "y.dat"), y, fmt='%.6f')

    with open(os.path.join(data_dir, "risk_before_after.csv",), 'a') as f:
        f.write(f'{classifier},{recourse},{risk_before:.3f},{risk_after:.3f}\n')
    
    if two_dim: 
        clf_dir = os.path.join(data_dir, classifier)
        os.makedirs(clf_dir, exist_ok=True)
        np.savetxt(os.path.join(clf_dir, "xyz_cond_probs.dat"), 
                xyz_cond_probs, fmt='%.6f')
        np.savetxt(os.path.join(clf_dir, "db_coords.dat"), 
                db_coords, fmt='%.6f')
        save_x_y_data(
            clf_dir, 
            X, 
            y, 
            predictions
            )

        recourse_dir = os.path.join(clf_dir, recourse)
        os.makedirs(recourse_dir, exist_ok=True)

        save_x_y_data(
            recourse_dir, 
            counterfactuals,
            y_after_recourse,
            predictions_after_recourse, 
            recourse=True
        )


def save_x_y_data(data_dir, X, y, y_hat, recourse=False):
    x_11, x_01, x_10, x_00 = tp_fn_fp_tn(X, y, y_hat)

    if not recourse:
        data_dir = os.path.join(data_dir, "original")
        os.makedirs(data_dir, exist_ok=True)

        np.savetxt(os.path.join(data_dir, "X.dat"), X, fmt='%.6f')

        np.savetxt(os.path.join(data_dir, "X_11.dat"), x_11, fmt='%.6f')
        np.savetxt(os.path.join(data_dir, "X_10.dat"), x_10, fmt='%.6f')
        np.savetxt(os.path.join(data_dir, "X_01.dat"), x_01, fmt='%.6f')
        np.savetxt(os.path.join(data_dir, "X_00.dat"), x_00, fmt='%.6f')

        np.savetxt(os.path.join(data_dir, "y.dat"), y, fmt='%.6f')
        np.savetxt(os.path.join(data_dir, "predictions.dat"), y_hat, fmt='%.6f')
    else:
        data_dir = os.path.join(data_dir, "recourse")
        os.makedirs(data_dir, exist_ok=True)

        np.savetxt(os.path.join(data_dir, "counterfactuals.dat"), X, fmt='%.6f')

        np.savetxt(os.path.join(data_dir, "X_11.dat"), x_11, fmt='%.6f')
        np.savetxt(os.path.join(data_dir, "X_10.dat"), x_10, fmt='%.6f')
        np.savetxt(os.path.join(data_dir, "X_01.dat"), x_01, fmt='%.6f')
        np.savetxt(os.path.join(data_dir, "X_00.dat"), x_00, fmt='%.6f')

        np.savetxt(os.path.join(data_dir, "y_after_recourse.dat"), y, fmt='%.6f')
        np.savetxt(os.path.join(data_dir, "predictions_after_recourse.dat"), y_hat, fmt='%.6f')


