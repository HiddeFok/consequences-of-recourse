"""
In this script all models that are used in the experiments are collected
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from models.recourse.brute_force_recourse import BruteForceRecourse
from models.recourse.wachter import Wachter
from models.recourse.growing_spheres import GrowingSpheres
from models.recourse.genetic_search import GeneticSearch

lr = LogisticRegression(class_weight='balanced')
ada = AdaBoostClassifier(n_estimators=10)
gbc = GradientBoostingClassifier(n_estimators=10)
tree = DecisionTreeClassifier(max_depth=4, class_weight='balanced')
gnb = GaussianNB()
rf = RandomForestClassifier(
    n_estimators=10, 
    max_depth=4, 
    class_weight='balanced'
)
qda = QuadraticDiscriminantAnalysis()
nn_1 = MLPClassifier(hidden_layer_sizes=(4,))
nn_2 = MLPClassifier(hidden_layer_sizes=(4, 4))
nn_3 = MLPClassifier(hidden_layer_sizes=(8,))
nn_4 = MLPClassifier(hidden_layer_sizes=(8, 16))
nn_5 = MLPClassifier(hidden_layer_sizes=(8, 16, 8))

models = {
    "lr": {"model": lr, "name": "Logistic Regression"},
    "gbc": {"model": gbc, "name": "Gradient Boosted Trees"},
    "tree": {"model": tree, "name": "Decision Tree"},
    "gnb": {"model": gnb, "name": "Gaussian Naive Bayes"}, 
    "rf": {"model": rf, "name": "Random Forest"},
    "qda": {"model": qda, "name": "QDA"},
    "NN_1": {"model":  nn_1, "name": "NN(4)"},
    "NN_2": {"model":  nn_2, "name": "NN(4, 4)"},
    "NN_3": {"model":  nn_3, "name": "NN(8)"},
    "NN_4": {"model":  nn_4, "name": "NN(8, 16)"},
    "NN_5": {"model":  nn_4, "name": "NN(8, 16, 8)"},
}

recourse_models = {
    'wachter': Wachter,
    'genetic_search': GeneticSearch, 
    'growing_spheres': GrowingSpheres,
}


recourse_models_synthetic = {
    'brute_force': BruteForceRecourse,
    'wachter': Wachter,
    'genetic_search': GeneticSearch, 
    'growing_spheres': GrowingSpheres,
}