"""
Utility functions to transform data into 
Latex (table) friendly formats

"""
import os
import pandas as pd
import numpy as np

from typing import TextIO, List


# Adjust these directories to point towards your own data
credit_dir = "checkpoints_cluster/credit_cluster"
adult_dir = "checkpoints_cluster/adult_cluster"
heloc_dir = "checkpoints_cluster/heloc_cluster"

suffix = "data/risk_before_after.csv"

col_names = ['clf', 'recourse', 'before', 'after']

clf_names = {
    'lr': "LR",
    'gbc': "GBT",
    'tree': 'DT',
    'gnb': 'NB',
    'qda': 'QDA', 
    'NN_1': 'NN(4)',
    'NN_2': 'NN(4, 4)', 
    'NN_3': 'NN(8)', 
    'NN_4': 'NN(8, 16)',
    'NN_5': 'NN(8, 16, 8)',
}
recourse_names = [
    'wachter', 
    'growing_spheres', 
    'genetic_search'
]

credit_all = []
adult_all = []
heloc_all = []
for i in range(10):
    credit_data_dir = os.path.join(credit_dir, f"experiment_{i+1}", suffix)
    adult_data_dir = os.path.join(adult_dir, f"experiment_{i+1}", suffix)
    heloc_data_dir = os.path.join(heloc_dir, f"experiment_{i+1}", suffix)

    credit = pd.read_csv(credit_data_dir, names=col_names, index_col=0)
    adult = pd.read_csv(adult_data_dir, names=col_names, index_col=0)
    heloc = pd.read_csv(heloc_data_dir, names=col_names, index_col=0)

    credit_all.append(credit)
    adult_all.append(adult)
    heloc_all.append(heloc)

credit = pd.concat(credit_all)
adult = pd.concat(adult_all)
heloc = pd.concat(heloc_all)

# Collecting data took too long to repeat it multiple times to create error bounds. A worst case approach is
# adopted, by using Popoviciu to upper bound the standard error. 
STD_BOUND = 0.5 / np.sqrt(1000)
NORMAL_975_QUANTILE = 1.96
T_976_QUANTILE = 2.26



def bold_before_after(before: float, after: float) -> str:
    if abs(before - after) < STD_BOUND:
        return f" & \\textbf{{{before:.2f}}} & \\textbf{{{after:.2f}}}"
    else:
        if before < after:
            return f" & \\textbf{{{before:.2f}}} & {after:.2f}"
        else:
            return f" & {before:.2f} & \\textbf{{{after:.2f}}}"


def bold_before_after_intervals(before: float, after: float, std_before: float, std_after) -> str:
    before_bound = NORMAL_975_QUANTILE * std_before
    after_bound = NORMAL_975_QUANTILE * std_after
    interval_before = (
        before - before_bound, 
        before + before_bound
    )
    interval_after = (
        after - after_bound, 
        after + after_bound
    )   

    intervals = sorted([interval_before, interval_after])
    
    if intervals[1][0] <= intervals[0][1]:
        string = (
            f"& \\textbf{{{before:.2f}}} $\\pm$ \\textbf{{{before_bound:.2f}}}"
            f"& \\textbf{{{after:.2f}}} $\\pm$ \\textbf{{{after_bound:.2f}}}\n"
        )
        return string
    else:
        if before < after:
            string = (
                f"& \\textbf{{{before:.2f}}} $\\pm$ \\textbf{{{before_bound:.2f}}}"
                f"& {after:.2f} $\\pm$ {after_bound:.2f}\n"
            )
            return string
        else:
            string = (
                f"& {before:.2f} $\\pm$ {before_bound:.2f}"
                f"& \\textbf{{{after:.2f}}} $\\pm$ \\textbf{{{after_bound:.2f}}}\n"
            )
            return string 


def write_line(f: TextIO, data: pd.DataFrame, recourse_names: List[str]) -> None:
    for rec in recourse_names:
        before = data[data["recourse"] == rec]["before"].values
        if len(before) == 0:
            before = 0
        else:
            before_mean = before.mean()
            before_std = before.std()
        after = data[data["recourse"] == rec]["after"].values
        if len(after) == 0:
            after = 0
        else:
            after_mean = after.mean()
            after_std = after.std()
        f.write(
            bold_before_after_intervals(
                before_mean, 
                after_mean,
                before_std,
                after_std
                )
            )


with open('data/real_data_table_credit.txt', 'w') as f:
    for clf in clf_names:
        f.write(f"{clf_names[clf]} ")
        credit_clf_data = credit.iloc[credit.index == clf]
        write_line(f, credit_clf_data, recourse_names)
        f.write("\\\\\n\n")

with open('data/real_data_table_adult.txt', 'w') as f:
    for clf in clf_names:
        f.write(f"{clf_names[clf]} ")
        adult_clf_data = adult.iloc[adult.index == clf]
        write_line(f, adult_clf_data, recourse_names)
        f.write("\\\\\n\n")

with open('data/real_data_table_heloc.txt', 'w') as f:
    for clf in clf_names:
        f.write(f"{clf_names[clf]} ")
        heloc_clf_data = heloc.iloc[heloc.index == clf]
        write_line(f, heloc_clf_data, recourse_names)
        f.write("\\\\\n\n")