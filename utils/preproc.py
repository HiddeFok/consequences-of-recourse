"""
Dataset pre-processing functions

1) Give me some Credit

2) Adult dataset

3) HELOC dataset

All numeric features are normalized to [0, 1] for simplicity 


"""
import os
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_moons, make_circles
from data.synthetic_data import create_gaussian_data


def stratified_sampling(
        data: pd.DataFrame,
        frac: float, 
        y_label: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
    stratified_data = data.groupby(y_label).apply(
        lambda x: x.sample(frac=frac)
    )
    X = stratified_data.loc[:, stratified_data.columns != y_label]
    y = stratified_data.loc[:, stratified_data.columns == y_label]
    return X, y


def data_splits(
        data: pd.DataFrame,
        fracs: Dict[str, float], 
        y_label: str
    ) -> Tuple[Dict, Dict]:
    Xs = {}
    ys = {}
    for name in fracs:
        X, y = stratified_sampling(data, fracs[name], y_label) 
        data = data.iloc[~data.index.isin(X.index), :]
        Xs[name], ys[name] = X.to_numpy(), y.values.ravel()
    return Xs, ys


def load_credit(
        fracs: Dict[str, float],
        dataset_folder: Union[str, bytes, os.PathLike] = "./data/csv_files"
    ) -> Dict:
    # Categorical features are:
    # NumberOfOpenCreditLinesAndLoans             
    # NumberOfTimes90DaysLate                     
    # NumberRealEstateLoansOrLines                
    # NumberOfTime60-89DaysPastDueNotWorse        
    # NumberOfDependents                          
    # age                                         
    # NumberOfTime30-59DaysPastDueNotWorse, 
    # But they should all have a monotonic relationship with the label, so we treat them as numeric

    data = pd.read_csv(
        os.path.join(dataset_folder, 'credit.csv'), 
        index_col=0,
        delimiter=","
    )

    data['label'] = data['SeriousDlqin2yrs']
    del data['SeriousDlqin2yrs']

    prop_pos = len(data[data['label'] == 1]) / len(data)
    print(f"Class Balance: (Positive, Negative) = ({prop_pos}, {1 - prop_pos})")

    feature_names = list(data.columns)
    feature_names.remove('label')

    scaler = MinMaxScaler()
    data[feature_names] = scaler.fit_transform(data[feature_names])

    Xs, ys = data_splits(data, fracs, y_label='label')

    return_dict = {
        'name': 'credit', 
        'best_class': 1, 
        'feature_names': feature_names, 
        'data': data, 
        'Xs': Xs, 
        'ys': ys, 
        'scaler': scaler
    }

    return return_dict


def load_adult(
        fracs: Dict[str, float],
        dataset_folder: Union[str, bytes, os.PathLike] = "./data/csv_files"
    ) -> Dict:
    data = pd.read_csv(
        os.path.join(dataset_folder, 'adult.csv'), 
        index_col=0,
        delimiter=","
    )

    # remove so-called useless columns as done by R. Guidotti 
    del data['fnlwgt']

    # There are no NA values again
    
    categorical_features = [
        'workclass_Private', 
        'marital-status_Non-Married', 
        'occupation_Other', 
        'race_White', 
        'sex_Male', 
        'native-country_US'
    ]

    # convert categorical features to codes
    for feat in categorical_features:
        data[feat] = pd.Categorical(data[feat])
        data[feat] = data[feat].cat.codes

    data['label'] = data['income']
    del data['income']

    prop_pos = len(data[data['label'] == 1]) / len(data)
    print(f"Class Balance: (Positive, Negative) = ({prop_pos}, {1 - prop_pos})")

    feature_names = list(data.columns)
    feature_names.remove('label')

    continuous_features = [feat for feat in feature_names if feat not in categorical_features]
    scaler = MinMaxScaler()
    data[continuous_features] = scaler.fit_transform(data[continuous_features])

    Xs, ys = data_splits(data, fracs, y_label='label')

    return_dict = {
        'name': 'credit', 
        'best_class': 1, 
        'feature_names': feature_names, 
        'categorical_features': categorical_features,
        'continuous_features': continuous_features,
        'data': data, 
        'Xs': Xs, 
        'ys': ys, 
        'scaler': scaler
    }

    return return_dict


def load_heloc(
        fracs: Dict[str, float],
        dataset_folder: Union[str, bytes, os.PathLike] = "./data/csv_files"
    ) -> Dict:
    data = pd.read_csv(
        os.path.join(dataset_folder, 'heloc.csv'), 
        index_col=0,
        delimiter=","
    )

    # There are no NA values again
    
    data['label'] = data['RiskPerformance']
    del data['RiskPerformance']

    prop_pos = len(data[data['label'] == 1]) / len(data)
    print(f"Class Balance: (Positive, Negative) = ({prop_pos}, {1 - prop_pos})")

    feature_names = list(data.columns)
    feature_names.remove('label')

    scaler = MinMaxScaler()
    data[feature_names] = scaler.fit_transform(data[feature_names])

    Xs, ys = data_splits(data, fracs, y_label='label')

    return_dict = {
        'name': 'credit', 
        'best_class': 1, 
        'feature_names': feature_names, 
        'data': data, 
        'Xs': Xs, 
        'ys': ys, 
        'scaler': scaler
    }

    return return_dict


load_funcs = {
    'credit': load_credit, 
    'adult': load_adult,
    'heloc': load_heloc
}

def load_data(
        dataset_name: str, 
        fracs: Dict[str, float], 
        dataset_folder: Union[str, bytes, os.PathLike] = './data/csv_files'
    ) -> Dict:
    load_func = load_funcs[dataset_name]
    return load_func(fracs, dataset_folder)


def make_gaussians(
        mean_1: np.array, 
        mean_2: np.array, 
        cov_1: np.array, 
        cov_2: np.array, 
        N: int
    ) -> Tuple[pd.DataFrame, pd.Series]:
    data = create_gaussian_data(mean_1, mean_2, cov_1, cov_2, N)
    X, y = data[:, :-1], data[:, -1]
    return X, y


load_synthetic_data_funcs = {
    'circles': make_circles, 
    'moons': make_moons,
    'gaussians': make_gaussians
}

def load_synthetic_data(
        dataset_name: str,
        *args, 
        **kwargs
    ) -> Tuple[pd.DataFrame, pd.Series]:
    load_func = load_synthetic_data_funcs[dataset_name]
    return load_func(*args, **kwargs)
