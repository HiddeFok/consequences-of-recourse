import os
import pandas as pd
from itertools import product

credit_dir = 'checkpoints_cluster/credit_cluster'
adult_dir = 'checkpoints_cluster/adult_cluster' 
heloc_dir = 'checkpoints_cluster/heloc_cluster'

suffix = "data/risk_before_after.csv"

clfs = ["lr", "gbc", "tree", "gnb", "rf", "qda", "NN_1", "NN_2", "NN_3", "NN_4", "NN_5"]
recourse_methods = ["wachter", "growing_spheres", "genetic_search"]
total_number = len(clfs) * len(recourse_methods)

print(f"Total number should be: {total_number}")

combinations = set(product(clfs, recourse_methods))

print('Missing data for credit experiment are:')
for i in range(10):
    data_dir = os.path.join(credit_dir, f"experiment_{i+1}", suffix)
    data = pd.read_csv(data_dir)
    succesful_experiments = set(data.iloc[:, 0:2].itertuples(index=False, name=None))
    missing = combinations - succesful_experiments
    print(f"\tNumber of missing in Experiment {i + 1} is {len(missing)}:")
    for j in missing:
        print(f"\t\t - {j}")

print('Missing data for heloc experiment are:')
for i in range(10):
    data_dir = os.path.join(heloc_dir, f"experiment_{i+1}", suffix)
    data = pd.read_csv(data_dir)
    succesful_experiments = set(data.iloc[:, 0:2].itertuples(index=False, name=None))
    missing = combinations - succesful_experiments
    print(f"\tNumber of missing in Experiment {i + 1} is {len(missing)}:")
    for j in missing:
        print(f"\t\t - {j}")

print('Missing data for adult experiment are:')
for i in range(10):
    data_dir = os.path.join(adult_dir, f"experiment_{i+1}", suffix)
    data = pd.read_csv(data_dir)
    succesful_experiments = set(data.iloc[:, 0:2].itertuples(index=False, name=None))
    missing = combinations - succesful_experiments
    print(f"\tNumber of missing in Experiment {i + 1} is {len(missing)}:")
    for j in missing:
        print(f"\t\t - {j}")

