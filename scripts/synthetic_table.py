"""
Utility functions to transform data into 
Latex (table) friendly formats

"""
import os
import numpy as np
import pandas as pd

from typing import TextIO, List


# Adjust these directories to point towards your own data
moons_dir = "checkpoints_cluster/moons_cluster"
circles_dir = "checkpoints_cluster/circles_cluster"
gaussians_dir = "checkpoints_cluster/gaussians_cluster"

suffix = "data/risk_before_after.csv"

col_names = ['clf', 'recourse', 'before', 'after']

clf_names = {
    'lr': "Logistic Regression (LR)",
    'gbc': "GradientBoostedTrees (GBT)",
    'tree': 'Decision Tree (DT)',
    'gnb': 'Naive Bayes (NB)',
    'qda': 'QuadraticDiscriminantAnalysis (QDA)', 
    'NN_1': 'Neural Network(4)',
    'NN_2': 'Neural Network(4, 4)', 
    'NN_3': 'Neural Network(8)', 
    'NN_4': 'Neural Network(8, 16)',
    'NN_5': 'Neural Netowrk(8, 16, 8)',
}
recourse_names = [
    'brute_force'
    # 'wachter', 
    # 'growing_spheres', 
    # 'genetic_search'
]

# loading in all the data
moons_all = []
circles_all = []
gaussians_all = []
for i in range(10):
    moons_data_dir = os.path.join(moons_dir, f"experiment_{i+1}", suffix)
    circles_data_dir = os.path.join(circles_dir, f"experiment_{i+1}", suffix)
    gaussians_data_dir = os.path.join(gaussians_dir, f"experiment_{i+1}", suffix)

    moons = pd.read_csv(moons_data_dir, names=col_names, index_col=0)
    circles = pd.read_csv(circles_data_dir, names=col_names, index_col=0)
    gaussians = pd.read_csv(gaussians_data_dir, names=col_names, index_col=0)

    moons_all.append(moons)
    circles_all.append(circles)
    gaussians_all.append(gaussians)


moons = pd.concat(moons_all)
circles = pd.concat(circles_all)
gaussians = pd.concat(gaussians_all)

STD_BOUND = 0.5 / np.sqrt(1000)
NORMAL_975_QUANTILE = 1.96
T_976_QUANTILE = 2.26


def bold_before_after(before: float, after: float, std_before: float, std_after) -> str:
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
            bold_before_after(
                before_mean, 
                after_mean, 
                before_std, 
                after_std
                )
            )

with open('data/synth_data_table.txt', 'w') as f:
    for clf in clf_names:
        f.write(f"{clf_names[clf]} \n")
        moons_clf_data = moons.iloc[moons.index == clf]
        write_line(f, moons_clf_data, recourse_names)

        circles_clf_data = circles.iloc[circles.index == clf]
        write_line(f, circles_clf_data, recourse_names)

        gaussians_clf_data = gaussians.iloc[gaussians.index == clf]
        write_line(f, gaussians_clf_data, recourse_names)
        f.write("\\\\\n\n")