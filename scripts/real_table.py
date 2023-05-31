"""
Utility functions to transform data into 
Latex (table) friendly formats

"""
import pandas as pd
import numpy as np

from typing import TextIO, List


# Adjust these directories to point towards your own data
credit_data_dir = "checkpoints/cluster_good_run/credit_cluster/data/risk_before_after.csv"
adult_data_dir = "checkpoints/cluster_good_run/adult_cluster/data/risk_before_after.csv"
heloc_data_dir = "checkpoints/cluster_good_run/heloc_cluster/data/risk_before_after.csv"

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

credit = pd.read_csv(credit_data_dir, names=col_names, index_col=0)
adult = pd.read_csv(adult_data_dir, names=col_names, index_col=0)
heloc = pd.read_csv(heloc_data_dir, names=col_names, index_col=0)


# Collecting data took too long to repeat it multiple times to create error bounds. A worst case approach is
# adopted, by using Popoviciu to upper bound the standard error. 
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


with open('data/real_data_table.txt', 'w') as f:
    for clf in clf_names:
        f.write(f"{clf_names[clf]} ")
        credit_clf_data = credit.iloc[credit.index == clf]
        write_line(f, credit_clf_data, recourse_names)
        f.write("\n")

        adult_clf_data = adult.iloc[adult.index == clf]
        write_line(f, adult_clf_data, recourse_names)
        f.write("\n")

        heloc_clf_data = heloc.iloc[heloc.index == clf]
        write_line(f, heloc_clf_data, recourse_names)
        f.write("\\\\\n\n")