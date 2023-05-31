"""
Utility functions to transform data into 
Latex (table) friendly formats

"""
import numpy as np
import pandas as pd

from typing import TextIO, List


# Adjust these directories to point towards your own data
moons_data_dir = "checkpoints/moons_cluster/data/risk_before_after.csv"
circles_data_dir = "checkpoints/circles_cluster/data/risk_before_after.csv"
gaussians_data_dir = "checkpoints/gaussians_cluster/data/risk_before_after.csv"

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

moons = pd.read_csv(moons_data_dir, names=col_names, index_col=0)
circles = pd.read_csv(circles_data_dir, names=col_names, index_col=0)
gaussians = pd.read_csv(gaussians_data_dir, names=col_names, index_col=0)

STD_BOUND = 0.5 / np.sqrt(1000)

def bold_before_after(before: float, after: float) -> str:
    if abs(before - after) < STD_BOUND:
        return f" & \\textbf{{{before:.2f}}} & \\textbf{{{after:.2f}}}"
    else:
        if before < after:
            return f" & \\textbf{{{before:.2f}}} & {after:.2f}"
        else:
            return f" & {before:.2f} & \\textbf{{{after:.2f}}}"


def write_line(f: TextIO, data: pd.DataFrame, recourse_names: List[str]) -> None:
    for rec in recourse_names:
        before = data[data["recourse"] == rec]["before"].values
        if len(before) == 0:
            before = 0
        else:
            before = before[0]
        after = data[data["recourse"] == rec]["after"].values
        if len(after) == 0:
            after = 0
        else:
            after = after[0]
        f.write(bold_before_after(before, after))


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